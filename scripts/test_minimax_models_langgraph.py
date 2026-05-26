"""Smoke-test MiniMax-M2.7 on the official China API through a minimal LangGraph.

MiniMax exposes ``MiniMax-M2.7`` through an OpenAI-compatible chat-completions
endpoint at ``https://api.minimaxi.com/v1``, so this script uses ``ChatOpenAI``
with that base URL and ``MINIMAX_CN_API_KEY``.
"""

from __future__ import annotations

import os
import sys
from typing import Annotated, Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

try:
    import tradingagents  # noqa: F401  # Loads local .env files when present.
except ImportError:
    pass


MINIMAX_CN_BASE_URL = "https://api.minimaxi.com/v1"


class State(TypedDict):
    messages: Annotated[list, add_messages]


def _extract_text(content: Any) -> str:
    """Handle both plain string content and LangChain block-list content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(text_parts) or str(content)
    return str(content)


def _format_api_error(exc: Exception) -> str:
    """Turn common MiniMax API failures into actionable messages."""
    message = str(exc)
    if "insufficient_balance_error" in message or "insufficient balance" in message:
        return (
            "MiniMax CN account has insufficient balance. "
            "Top up at https://platform.minimaxi.com/ — the LangGraph + ChatOpenAI "
            "setup reached the API successfully."
        )
    if "401" in message or "authentication" in message.lower():
        return (
            "Authentication failed. Check MINIMAX_CN_API_KEY in your shell config "
            "(~/.zshrc) and reload with `source ~/.zshrc`."
        )
    return message


def test_model_with_langgraph(model_name: str) -> bool:
    print(f"\n{'=' * 50}")
    print(f"Testing model: {model_name}")
    print(f"{'=' * 50}")

    api_key = os.environ.get("MINIMAX_CN_API_KEY")
    if not api_key:
        print("Error: MINIMAX_CN_API_KEY environment variable is not set.")
        return False

    llm = ChatOpenAI(
        model=model_name,
        base_url=MINIMAX_CN_BASE_URL,
        api_key=api_key,
        temperature=0,
        # Keep reasoning blocks out of message.content on native MiniMax API.
        extra_body={"reasoning_split": True},
    )

    def call_model(state: State):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    workflow = StateGraph(State)
    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
    app = workflow.compile()

    test_message = HumanMessage(
        content="Hello! Please reply with exactly 'API is working' and nothing else."
    )
    print(f"Sending message to LangGraph node using {model_name}...")

    try:
        result = app.invoke({"messages": [test_message]})
        text = _extract_text(result["messages"][-1].content).strip()
        print(f"Response: {text}")
        if text == "API is working":
            print(f"Success: {model_name} is working in LangGraph.")
            return True
        print(f"Warning: request succeeded, but the response did not match exactly.")
        return False
    except Exception as exc:
        print(f"Error testing {model_name}: {_format_api_error(exc)}")
        return False


def main() -> int:
    print("Testing MiniMax China API through LangGraph")
    print(f"Base URL: {MINIMAX_CN_BASE_URL}")

    if not os.environ.get("MINIMAX_CN_API_KEY"):
        print("Warning: MINIMAX_CN_API_KEY environment variable is not set.")
        print("The API calls will likely fail.")

    models_to_test = [
        "MiniMax-M2.7",
    ]

    results = [test_model_with_langgraph(model) for model in models_to_test]
    return 0 if all(results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
