"""Shared helpers for invoking an agent with structured output and a graceful fallback.

The Asset Manager, Trader, and Research Manager all follow the same
canonical pattern:

1. At agent creation, wrap the LLM with ``with_structured_output(Schema)``
   so the model returns a typed Pydantic instance. If the provider does
   not support structured output (rare; mostly older Ollama models), the
   wrap is skipped and the agent uses free-text generation instead.
2. At invocation, run the structured call and render the result back to
   markdown. If the structured call itself fails for any reason
   (malformed JSON from a weak model, transient provider issue), fall
   back to a plain ``llm.invoke`` so the pipeline never blocks.

Centralising the pattern here keeps the agent factories small and ensures
all three agents log the same warnings when fallback fires.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_TOOL_NUDGE = (
    "You MUST respond by calling the {schema_name} tool with valid structured "
    "arguments. Do not reply in free text or markdown."
)


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Optional[Any]:
    """Return ``llm.with_structured_output(schema)`` or ``None`` if unsupported.

    Logs a warning when the binding fails so the user understands the agent
    will use free-text generation for every call instead of one-shot fallback.
    """
    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name, exc,
        )
        return None


def invoke_structured_or_freetext(
    structured_llm: Optional[Any],
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
) -> str:
    """Run the structured call and render to markdown; fall back to free-text on any failure.

    ``prompt`` is whatever the underlying LLM accepts (a string for chat
    invocations, a list of message dicts for chat models that take that
    shape). The same value is forwarded to the free-text path so the
    fallback sees the same input the structured call did.
    """
    if structured_llm is not None:
        try:
            result = structured_llm.invoke(prompt)
            return render(result)
        except Exception as exc:
            logger.warning(
                "%s: structured-output invocation failed (%s); retrying once as free text",
                agent_name, exc,
            )

    response = plain_llm.invoke(prompt)
    return response.content


def _validate_schema(schema: type[T], data: Any) -> Optional[T]:
    try:
        return schema.model_validate(data)
    except (ValidationError, TypeError, ValueError):
        return None


def _tool_call_args_from_message(raw: Any) -> list[dict]:
    """Extract tool-call argument dicts from a LangChain AIMessage."""
    if raw is None:
        return []

    collected: list[dict] = []
    for tc in getattr(raw, "tool_calls", None) or []:
        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
        if isinstance(args, dict):
            collected.append(args)

    if collected:
        return collected

    additional = getattr(raw, "additional_kwargs", None) or {}
    for tc in additional.get("tool_calls", []) or []:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        arguments = fn.get("arguments")
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                collected.append(parsed)
        elif isinstance(arguments, dict):
            collected.append(arguments)

    return collected


def _parse_json_from_content(content: Any, schema: type[T]) -> Optional[T]:
    """Try to parse a JSON object embedded in message content."""
    if not content:
        return None
    text = content if isinstance(content, str) else str(content)
    text = text.strip()
    if not text:
        return None

    candidates = [text]
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidates.insert(0, fence.group(1))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            inst = _validate_schema(schema, data)
            if inst is not None:
                return inst
    return None


def _coerce_structured_result(
    output: Any,
    schema: type[T],
    agent_name: str,
) -> Optional[T]:
    """Normalise a with_structured_output(include_raw=True) result to schema."""
    if isinstance(output, schema):
        return output

    if not isinstance(output, dict):
        return None

    parsed = output.get("parsed")
    if isinstance(parsed, schema):
        return parsed

    raw = output.get("raw")
    for args in _tool_call_args_from_message(raw):
        inst = _validate_schema(schema, args)
        if inst is not None:
            logger.info(
                "%s: recovered structured output from tool-call arguments",
                agent_name,
            )
            return inst

    if raw is not None:
        inst = _parse_json_from_content(getattr(raw, "content", ""), schema)
        if inst is not None:
            logger.info(
                "%s: recovered structured output from message content JSON",
                agent_name,
            )
            return inst

    parsing_error = output.get("parsing_error")
    if parsing_error is not None:
        logger.warning(
            "%s: structured-output parsing error (%s)",
            agent_name,
            parsing_error,
        )

    return None


def invoke_structured_model(
    llm: Any,
    schema: type[T],
    prompt: Any,
    agent_name: str,
) -> Optional[T]:
    """Invoke structured output with MiniMax-safe recovery and one retry.

    Uses ``include_raw=True`` so we can salvage tool-call arguments when
    LangChain's parser returns ``parsed=None`` (common with MiniMax M2.x
    when the model emits reasoning in ``content`` alongside a valid tool
    call). Retries once with an explicit tool-only nudge if the first
    attempt yields nothing parseable.
    """
    try:
        structured = llm.with_structured_output(schema, include_raw=True)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s)",
            agent_name,
            exc,
        )
        return None

    nudge = _TOOL_NUDGE.format(schema_name=schema.__name__)

    for attempt, current_prompt in enumerate((prompt, f"{prompt}\n\n{nudge}"), start=1):
        try:
            output = structured.invoke(current_prompt)
        except Exception as exc:
            logger.warning(
                "%s: structured-output invocation failed on attempt %d (%s)",
                agent_name,
                attempt,
                exc,
            )
            continue

        result = _coerce_structured_result(output, schema, agent_name)
        if result is not None:
            return result

        if attempt == 1:
            logger.warning(
                "%s: structured output empty on attempt 1; retrying with tool nudge",
                agent_name,
            )

    return None
