"""Smoke-test MiniMax M2.7 on NVIDIA NIM through a minimal LangGraph.

NVIDIA exposes ``minimaxai/minimax-m2.7`` through an OpenAI-compatible
chat-completions endpoint, so this script uses ``ChatOpenAI`` with NVIDIA's
base URL. ``ChatGoogleGenerativeAI`` only supports Google's Gemini API and
cannot call NVIDIA NIM endpoints.
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


MODEL_NAME = "minimaxai/minimax-m2.7"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


class State(TypedDict):
    messages: Annotated[list, add_messages]


def _api_key() -> str | None:
    """Prefer OpenAI-compatible naming, then fall back to NVIDIA-specific env."""
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("NVIDIA_API_KEY")


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


def build_graph():
    key = _api_key()
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY or NVIDIA_API_KEY before running this script.")

    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=NVIDIA_BASE_URL,
        api_key=key,
        # 0 = greedy decoding; helps the smoke test match the exact reply string.
        temperature=0,
    )

    def call_model(state: State):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    workflow = StateGraph(State)
    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
    return workflow.compile()


def main() -> int:
    print(f"Testing {MODEL_NAME} on NVIDIA NIM through LangGraph")
    print(f"Base URL: {NVIDIA_BASE_URL}")

    app = build_graph()
    test_message = HumanMessage(
        content="Hello! Please reply with exactly 'API is working' and nothing else."
    )

    result = app.invoke({"messages": [test_message]})
    final_message = result["messages"][-1]
    text = _extract_text(final_message.content).strip()

    print(f"Response: {text}")
    if text == "API is working":
        print("Success: NVIDIA MiniMax M2.7 is working in LangGraph.")
        return 0

    print("Warning: request succeeded, but the response did not match exactly.")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
