# TradingAgents/graph/portfolio_setup.py

from typing import Any

from langgraph.graph import END, START, StateGraph

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.utils.agent_states import PortfolioState


class PortfolioGraphSetup:
    """Handles setup of the portfolio-level trading graph."""

    def __init__(self, deep_thinking_llm: Any):
        self.deep_thinking_llm = deep_thinking_llm

    def setup_graph(self):
        """Build a single-node graph: Portfolio Manager."""
        portfolio_manager_node = create_portfolio_manager(self.deep_thinking_llm)

        workflow = StateGraph(PortfolioState)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)
        workflow.add_edge(START, "Portfolio Manager")
        workflow.add_edge("Portfolio Manager", END)

        return workflow
