"""Tests for structured-output recovery (MiniMax M2.x tool-call salvage)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from tradingagents.agents.utils.structured import (
    _coerce_structured_result,
    _parse_json_from_content,
    _tool_call_args_from_message,
    invoke_structured_model,
)


class _SamplePlan(BaseModel):
    label: str
    count: int = 0


@pytest.mark.unit
class TestToolCallArgsFromMessage:
    def test_reads_langchain_tool_calls(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "_SamplePlan", "args": {"label": "ok", "count": 2}, "id": "1"}],
        )
        assert _tool_call_args_from_message(msg) == [{"label": "ok", "count": 2}]

    def test_reads_openai_style_additional_kwargs(self):
        msg = AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "function": {
                            "name": "_SamplePlan",
                            "arguments": '{"label": "from_kwargs", "count": 1}',
                        }
                    }
                ]
            },
        )
        assert _tool_call_args_from_message(msg) == [{"label": "from_kwargs", "count": 1}]


@pytest.mark.unit
class TestCoerceStructuredResult:
    def test_parsed_dict_path(self):
        plan = _SamplePlan(label="direct", count=3)
        assert _coerce_structured_result(plan, _SamplePlan, "test") == plan

    def test_recover_from_include_raw_tool_calls(self):
        raw = AIMessage(
            content="some reasoning prose",
            tool_calls=[{"name": "_SamplePlan", "args": {"label": "recovered"}, "id": "1"}],
        )
        out = _coerce_structured_result(
            {"parsed": None, "raw": raw, "parsing_error": None},
            _SamplePlan,
            "test",
        )
        assert out == _SamplePlan(label="recovered")

    def test_recover_json_from_content(self):
        raw = AIMessage(content='{"label": "json", "count": 5}')
        out = _coerce_structured_result(
            {"parsed": None, "raw": raw, "parsing_error": None},
            _SamplePlan,
            "test",
        )
        assert out == _SamplePlan(label="json", count=5)


@pytest.mark.unit
class TestParseJsonFromContent:
    def test_parses_fenced_json(self):
        content = 'Here is the plan:\n```json\n{"label": "fenced", "count": 7}\n```'
        assert _parse_json_from_content(content, _SamplePlan) == _SamplePlan(
            label="fenced", count=7
        )


@pytest.mark.unit
class TestInvokeStructuredModel:
    def test_returns_parsed_instance(self):
        plan = _SamplePlan(label="happy")
        structured = MagicMock()
        structured.invoke.return_value = {
            "parsed": plan,
            "raw": AIMessage(content=""),
            "parsing_error": None,
        }
        llm = MagicMock()
        llm.with_structured_output.return_value = structured

        result = invoke_structured_model(llm, _SamplePlan, "prompt", "Agent")
        assert result == plan
        llm.with_structured_output.assert_called_once_with(_SamplePlan, include_raw=True)

    def test_retries_with_nudge_when_first_attempt_empty(self):
        plan = _SamplePlan(label="retry")
        structured = MagicMock()
        structured.invoke.side_effect = [
            {"parsed": None, "raw": AIMessage(content="prose only"), "parsing_error": None},
            {
                "parsed": plan,
                "raw": AIMessage(content=""),
                "parsing_error": None,
            },
        ]
        llm = MagicMock()
        llm.with_structured_output.return_value = structured

        result = invoke_structured_model(llm, _SamplePlan, "prompt", "Agent")
        assert result == plan
        assert structured.invoke.call_count == 2
