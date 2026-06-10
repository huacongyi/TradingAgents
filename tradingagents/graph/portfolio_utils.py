"""Utilities for loading per-ticker analysis logs and enforcing portfolio constraints."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from tradingagents.agents.schemas import (
    PortfolioOrder,
    PortfolioOrderSide,
    PortfolioTradingPlan,
    WorkingOrderAction,
)
from tradingagents.agents.utils.rating import parse_rating
from tradingagents.dataflows.utils import safe_ticker_component

logger = logging.getLogger(__name__)


def _order_notional(order: Dict[str, Any]) -> float:
    """Return cash reserved by a working order (BUY only)."""
    if str(order.get("side", "")).upper() != "BUY":
        return 0.0
    return float(order.get("quantity", 0)) * float(order.get("price", 0))


def _portfolio_holdings(portfolio: List[Dict[str, Any]]) -> Dict[str, int]:
    holdings: Dict[str, int] = {}
    for position in portfolio:
        ticker = position["ticker_symbol"]
        qty = int(float(position.get("quantity", 0)))
        holdings[ticker] = holdings.get(ticker, 0) + qty
    return holdings


def load_ticker_decisions(
    results_dir: str,
    symbols: Iterable[str],
    trade_date: str,
) -> List[Dict[str, Any]]:
    """Load per-ticker final_trade_decision from analysis logs for trade_date."""
    decisions: List[Dict[str, Any]] = []
    base = Path(results_dir)

    for symbol in symbols:
        safe_ticker = safe_ticker_component(symbol)
        log_path = (
            base
            / safe_ticker
            / "TradingAgentsStrategy_logs"
            / f"full_states_log_{trade_date}.json"
        )
        if not log_path.is_file():
            logger.warning(
                "No analysis log for %s on %s at %s; skipping",
                symbol,
                trade_date,
                log_path,
            )
            continue

        with open(log_path, encoding="utf-8") as f:
            data = json.load(f)

        company = data.get("company_of_interest", symbol)
        final_decision = data.get("final_trade_decision", "")
        decisions.append({
            "company_of_interest": company,
            "final_trade_decision": final_decision,
            "rating": parse_rating(final_decision),
        })

    return decisions


def compute_cash_budget(
    account_balances: Dict[str, Any],
    working_orders: List[Dict[str, Any]],
    working_order_actions: List[WorkingOrderAction],
) -> float:
    """Return cash available for new BUY orders after reconciling working orders."""
    cash = float(account_balances.get("cash", 0))
    action_by_ticker = {a.ticker_symbol: a for a in working_order_actions}
    working_tickers: Set[str] = {o["ticker_symbol"] for o in working_orders}

    for order in working_orders:
        ticker = order["ticker_symbol"]
        notional = _order_notional(order)
        action = action_by_ticker.get(ticker)

        if action is None:
            # No explicit action — treat as KEEP (cash remains reserved).
            cash -= notional
            continue

        if action.action == "KEEP":
            cash -= notional
        elif action.action == "CANCEL":
            pass  # frees reserved cash
        elif action.action == "REPLACE":
            if action.replacement and action.replacement.side == PortfolioOrderSide.BUY:
                cash -= action.replacement.quantity * action.replacement.price
            # original working order notional is released

    # Actions for tickers without a matching working order are ignored.
    for ticker, action in action_by_ticker.items():
        if ticker not in working_tickers:
            logger.warning(
                "Working order action for %s has no matching working order; ignoring",
                ticker,
            )

    return max(cash, 0.0)


def _validate_working_order_actions(
    working_orders: List[Dict[str, Any]],
    actions: List[WorkingOrderAction],
) -> List[WorkingOrderAction]:
    """Drop invalid actions and log warnings."""
    working_tickers = {o["ticker_symbol"] for o in working_orders}
    valid: List[WorkingOrderAction] = []

    for action in actions:
        if action.ticker_symbol not in working_tickers:
            logger.warning(
                "Dropping working order action for unknown ticker %s",
                action.ticker_symbol,
            )
            continue
        if action.action == "REPLACE" and action.replacement is None:
            logger.warning(
                "REPLACE action for %s missing replacement; treating as CANCEL",
                action.ticker_symbol,
            )
            valid.append(
                WorkingOrderAction(ticker_symbol=action.ticker_symbol, action="CANCEL")
            )
            continue
        valid.append(action)

    return valid


def _clamp_buy_orders(
    orders: List[PortfolioOrder],
    cash_budget: float,
) -> List[PortfolioOrder]:
    """Drop or shrink BUY orders until total notional fits cash_budget."""
    result: List[PortfolioOrder] = []
    remaining = cash_budget

    for order in orders:
        if order.side != PortfolioOrderSide.BUY:
            result.append(order)
            continue

        notional = order.quantity * order.price
        if notional <= remaining:
            result.append(order)
            remaining -= notional
            continue

        max_qty = int(remaining // order.price) if order.price > 0 else 0
        if max_qty >= 1:
            logger.warning(
                "Clamping BUY %s from %d to %d shares to fit cash budget",
                order.ticker_symbol,
                order.quantity,
                max_qty,
            )
            result.append(order.model_copy(update={"quantity": max_qty}))
            remaining -= max_qty * order.price
        else:
            logger.warning(
                "Dropping BUY %s (%d shares) — insufficient cash",
                order.ticker_symbol,
                order.quantity,
            )

    return result


def _clamp_sell_orders(
    orders: List[PortfolioOrder],
    holdings: Dict[str, int],
) -> List[PortfolioOrder]:
    """Clamp SELL quantities to current holdings."""
    result: List[PortfolioOrder] = []

    for order in orders:
        if order.side != PortfolioOrderSide.SELL:
            result.append(order)
            continue

        held = holdings.get(order.ticker_symbol, 0)
        if held <= 0:
            logger.warning(
                "Dropping SELL %s — no shares held",
                order.ticker_symbol,
            )
            continue

        qty = min(order.quantity, held)
        if qty < order.quantity:
            logger.warning(
                "Clamping SELL %s from %d to %d shares (held=%d)",
                order.ticker_symbol,
                order.quantity,
                qty,
                held,
            )
        result.append(order.model_copy(update={"quantity": qty}))

    return result


def enforce_portfolio_constraints(
    plan: PortfolioTradingPlan,
    portfolio: List[Dict[str, Any]],
    working_orders: List[Dict[str, Any]],
    account_balances: Dict[str, Any],
) -> PortfolioTradingPlan:
    """Apply deterministic cash and position limits to the LLM plan."""
    working_order_actions = _validate_working_order_actions(
        working_orders, plan.working_order_actions
    )
    cash_budget = compute_cash_budget(
        account_balances, working_orders, working_order_actions
    )
    holdings = _portfolio_holdings(portfolio)

    orders = [
        o for o in plan.orders
        if o.quantity >= 1 and o.price > 0
    ]
    orders = _clamp_sell_orders(orders, holdings)
    orders = _clamp_buy_orders(orders, cash_budget)

    # Validate replacement orders in working_order_actions
    validated_actions: List[WorkingOrderAction] = []
    for action in working_order_actions:
        if action.action != "REPLACE" or action.replacement is None:
            validated_actions.append(action)
            continue

        replacement = action.replacement
        if replacement.side == PortfolioOrderSide.SELL:
            held = holdings.get(replacement.ticker_symbol, 0)
            qty = min(replacement.quantity, held)
            if qty < 1:
                logger.warning(
                    "REPLACE for %s: replacement SELL invalid (no shares); CANCEL instead",
                    action.ticker_symbol,
                )
                validated_actions.append(
                    WorkingOrderAction(
                        ticker_symbol=action.ticker_symbol,
                        action="CANCEL",
                    )
                )
                continue
            if qty < replacement.quantity:
                replacement = replacement.model_copy(update={"quantity": qty})
        validated_actions.append(
            WorkingOrderAction(
                ticker_symbol=action.ticker_symbol,
                action="REPLACE",
                replacement=replacement,
            )
        )

    return PortfolioTradingPlan(
        executive_summary=plan.executive_summary,
        working_order_actions=validated_actions,
        orders=orders,
    )


def plan_to_dict(plan: PortfolioTradingPlan) -> Dict[str, Any]:
    """Serialize a PortfolioTradingPlan for JSON logging."""
    return plan.model_dump(mode="json")
