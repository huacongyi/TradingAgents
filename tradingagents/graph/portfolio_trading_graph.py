# TradingAgents/graph/portfolio_trading_graph.py

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from tradingagents.agents.schemas import (
    PortfolioTradingPlan,
    render_portfolio_trading_plan,
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.dataflows.config import set_config
from tradingagents.llm_clients import create_llm_client

from .portfolio_setup import PortfolioGraphSetup
from .portfolio_utils import (
    enforce_portfolio_constraints,
    load_ticker_decisions,
    plan_to_dict,
)

logger = logging.getLogger(__name__)


class PortfolioTradingGraph:
    """Orchestrates portfolio-level trading decisions from per-ticker analysis logs."""

    def __init__(
        self,
        debug: bool = False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
    ):
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []

        set_config(self.config)
        os.makedirs(self.config["results_dir"], exist_ok=True)

        llm_kwargs = self._get_provider_kwargs()
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        self.deep_thinking_llm = deep_client.get_llm()

        self.graph_setup = PortfolioGraphSetup(self.deep_thinking_llm)
        self.workflow = self.graph_setup.setup_graph()
        self.graph = self.workflow.compile()

    def _get_provider_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level
        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
        elif provider == "anthropic":
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

        return kwargs

    def propagate(
        self,
        trade_date: str,
        symbols: Union[Set[str], List[str]],
        portfolio: List[Dict[str, Any]],
        working_orders: List[Dict[str, Any]],
        account_balances: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], PortfolioTradingPlan]:
        """Run the portfolio graph and return the enforced trading plan."""
        symbol_list = sorted(symbols)
        ticker_decisions = load_ticker_decisions(
            self.config["results_dir"],
            symbol_list,
            str(trade_date),
        )

        init_state = {
            "trade_date": str(trade_date),
            "symbols": symbol_list,
            "portfolio": portfolio,
            "working_orders": working_orders,
            "account_balances": account_balances,
            "ticker_decisions": ticker_decisions,
            "portfolio_trading_plan": "",
            "structured_plan": None,
        }

        args = {
            "stream_mode": ["updates", "values"],
            "config": {"recursion_limit": self.config.get("max_recur_limit", 100)},
        }

        final_state: Dict[str, Any] = {}
        for mode, chunk in self.graph.stream(init_state, **args):
            if mode == "updates":
                for node_name, node_delta in chunk.items():
                    if not self.debug or not isinstance(node_delta, dict):
                        continue
                    if node_name == "Portfolio Manager":
                        plan_text = node_delta.get("portfolio_trading_plan")
                        if plan_text:
                            print("================================== Portfolio Plan ==================================")
                            print()
                            print(plan_text)
                continue
            final_state = chunk

        raw_plan = final_state.get("structured_plan")
        if raw_plan is None or not isinstance(raw_plan, PortfolioTradingPlan):
            raw_plan = PortfolioTradingPlan(
                executive_summary=final_state.get("portfolio_trading_plan", ""),
                working_order_actions=[],
                orders=[],
            )

        enforced_plan = enforce_portfolio_constraints(
            raw_plan,
            portfolio,
            working_orders,
            account_balances,
        )

        final_state["structured_plan"] = enforced_plan
        final_state["portfolio_trading_plan"] = render_portfolio_trading_plan(
            enforced_plan
        )

        self._log_plan(
            trade_date=str(trade_date),
            symbols=symbol_list,
            portfolio=portfolio,
            working_orders=working_orders,
            account_balances=account_balances,
            ticker_decisions=ticker_decisions,
            plan=enforced_plan,
        )

        return final_state, enforced_plan

    def _log_plan(
        self,
        trade_date: str,
        symbols: List[str],
        portfolio: List[Dict[str, Any]],
        working_orders: List[Dict[str, Any]],
        account_balances: Dict[str, Any],
        ticker_decisions: List[Dict[str, Any]],
        plan: PortfolioTradingPlan,
    ) -> None:
        directory = (
            Path(self.config["results_dir"])
            / "portfolio"
            / "PortfolioTrading_logs"
        )
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"portfolio_plan_{trade_date}.json"
        payload = {
            "trade_date": trade_date,
            "symbols": symbols,
            "portfolio": portfolio,
            "working_orders": working_orders,
            "account_balances": account_balances,
            "ticker_decisions": [
                {
                    "company_of_interest": d["company_of_interest"],
                    "rating": d.get("rating"),
                    "final_trade_decision": d.get("final_trade_decision"),
                }
                for d in ticker_decisions
            ],
            "plan": plan_to_dict(plan),
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

        logger.info("Portfolio plan saved to %s", log_path)
