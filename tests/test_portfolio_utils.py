"""Unit tests for portfolio log loading and constraint enforcement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradingagents.agents.schemas import (
    PortfolioOrder,
    PortfolioOrderSide,
    PortfolioTradingPlan,
    WorkingOrderAction,
)
from tradingagents.graph.portfolio_utils import (
    compute_cash_budget,
    enforce_portfolio_constraints,
    load_ticker_decisions,
)


@pytest.fixture
def sample_log_dir(tmp_path: Path) -> Path:
    ticker_dir = tmp_path / "NVDA" / "TradingAgentsStrategy_logs"
    ticker_dir.mkdir(parents=True)
    payload = {
        "company_of_interest": "NVDA",
        "final_trade_decision": "**Rating**: Hold\n\n**Executive Summary**: Stay flat.",
    }
    (ticker_dir / "full_states_log_2026-06-09.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return tmp_path


@pytest.mark.unit
class TestLoadTickerDecisions:
    def test_loads_existing_log(self, sample_log_dir: Path):
        decisions = load_ticker_decisions(
            str(sample_log_dir), ["NVDA"], "2026-06-09"
        )
        assert len(decisions) == 1
        assert decisions[0]["company_of_interest"] == "NVDA"
        assert decisions[0]["rating"] == "Hold"

    def test_skips_missing_log(self, sample_log_dir: Path):
        decisions = load_ticker_decisions(
            str(sample_log_dir), ["NVDA", "MISSING"], "2026-06-09"
        )
        assert len(decisions) == 1
        assert decisions[0]["company_of_interest"] == "NVDA"


@pytest.mark.unit
class TestComputeCashBudget:
    def test_keep_reserves_cash(self):
        working_orders = [
            {"ticker_symbol": "AMZN", "side": "BUY", "quantity": 1, "price": 230.0},
        ]
        actions = [
            WorkingOrderAction(ticker_symbol="AMZN", action="KEEP"),
        ]
        budget = compute_cash_budget(
            {"cash": 1000.0},
            working_orders,
            actions,
        )
        assert budget == pytest.approx(770.0)

    def test_cancel_frees_cash(self):
        working_orders = [
            {"ticker_symbol": "AMZN", "side": "BUY", "quantity": 1, "price": 230.0},
        ]
        actions = [
            WorkingOrderAction(ticker_symbol="AMZN", action="CANCEL"),
        ]
        budget = compute_cash_budget(
            {"cash": 1000.0},
            working_orders,
            actions,
        )
        assert budget == pytest.approx(1000.0)

    def test_replace_swaps_notional(self):
        working_orders = [
            {"ticker_symbol": "UEC", "side": "BUY", "quantity": 5, "price": 10.0},
        ]
        replacement = PortfolioOrder(
            ticker_symbol="UEC",
            side=PortfolioOrderSide.BUY,
            quantity=2,
            price=12.0,
        )
        actions = [
            WorkingOrderAction(
                ticker_symbol="UEC",
                action="REPLACE",
                replacement=replacement,
            ),
        ]
        budget = compute_cash_budget(
            {"cash": 1000.0},
            working_orders,
            actions,
        )
        # 1000 - 24 (replacement) = 976; original 50 freed
        assert budget == pytest.approx(976.0)


@pytest.mark.unit
class TestEnforcePortfolioConstraints:
    def test_clamps_oversized_buy(self):
        plan = PortfolioTradingPlan(
            executive_summary="Buy too much",
            working_order_actions=[],
            orders=[
                PortfolioOrder(
                    ticker_symbol="TSLA",
                    side=PortfolioOrderSide.BUY,
                    quantity=100,
                    price=400.0,
                ),
            ],
        )
        enforced = enforce_portfolio_constraints(
            plan,
            portfolio=[],
            working_orders=[],
            account_balances={"cash": 1000.0},
        )
        assert len(enforced.orders) == 1
        assert enforced.orders[0].quantity == 2  # 2 * 400 = 800 <= 1000

    def test_drops_sell_without_holdings(self):
        plan = PortfolioTradingPlan(
            executive_summary="Sell TSLA",
            working_order_actions=[],
            orders=[
                PortfolioOrder(
                    ticker_symbol="TSLA",
                    side=PortfolioOrderSide.SELL,
                    quantity=5,
                    price=400.0,
                ),
            ],
        )
        enforced = enforce_portfolio_constraints(
            plan,
            portfolio=[{"ticker_symbol": "NVDA", "quantity": 3, "trade_price": 200}],
            working_orders=[],
            account_balances={"cash": 5000.0},
        )
        assert enforced.orders == []

    def test_clamps_sell_to_holdings(self):
        plan = PortfolioTradingPlan(
            executive_summary="Trim NVDA",
            working_order_actions=[],
            orders=[
                PortfolioOrder(
                    ticker_symbol="NVDA",
                    side=PortfolioOrderSide.SELL,
                    quantity=10,
                    price=200.0,
                ),
            ],
        )
        enforced = enforce_portfolio_constraints(
            plan,
            portfolio=[{"ticker_symbol": "NVDA", "quantity": 3, "trade_price": 200}],
            working_orders=[],
            account_balances={"cash": 5000.0},
        )
        assert len(enforced.orders) == 1
        assert enforced.orders[0].quantity == 3
