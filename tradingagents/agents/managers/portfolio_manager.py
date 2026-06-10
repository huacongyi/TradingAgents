"""Portfolio Manager: synthesises per-ticker decisions into a portfolio trading plan."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from tradingagents.agents.schemas import (
    PortfolioTradingPlan,
    render_portfolio_trading_plan,
)
from tradingagents.agents.utils.agent_utils import get_language_instruction
from tradingagents.agents.utils.structured import invoke_structured_model

logger = logging.getLogger(__name__)

# Keep prompts bounded — full Asset Manager logs can be very long.
_MAX_DECISION_CHARS = 2000
_EXEC_SUMMARY_RE = re.compile(
    r"\*\*Executive Summary\*\*:\s*(.+?)(?:\n\n|\*\*|$)",
    re.DOTALL | re.IGNORECASE,
)


def _extract_decision_excerpt(decision: str) -> str:
    """Prefer the executive summary; fall back to a bounded excerpt."""
    match = _EXEC_SUMMARY_RE.search(decision)
    if match:
        text = match.group(1).strip()
    else:
        text = decision.strip()
    if len(text) > _MAX_DECISION_CHARS:
        return text[:_MAX_DECISION_CHARS] + "\n...(truncated)..."
    return text


def _format_ticker_decisions(decisions: List[Dict[str, Any]]) -> str:
    if not decisions:
        return "(No per-ticker analysis logs available.)"

    parts: List[str] = []
    for entry in decisions:
        ticker = entry["company_of_interest"]
        rating = entry.get("rating", "Hold")
        excerpt = _extract_decision_excerpt(entry.get("final_trade_decision", ""))
        parts.append(
            f"### {ticker} (Rating: {rating})\n{excerpt}\n"
        )
    return "\n".join(parts)


def _fallback_plan(content: str) -> PortfolioTradingPlan:
    return PortfolioTradingPlan(
        executive_summary=content,
        working_order_actions=[],
        orders=[],
    )


def _invoke_portfolio_plan(
    llm,
    prompt: str,
) -> PortfolioTradingPlan:
    """Invoke structured output with MiniMax recovery; fall back to free text."""
    plan = invoke_structured_model(
        llm,
        PortfolioTradingPlan,
        prompt,
        "Portfolio Manager",
    )
    if plan is not None:
        return plan

    logger.warning(
        "Portfolio Manager: structured output unavailable; retrying as free text",
    )
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    return _fallback_plan(content)


def _format_portfolio(portfolio: List[Dict[str, Any]]) -> str:
    if not portfolio:
        return "(No current holdings.)"
    lines = []
    for pos in portfolio:
        lines.append(
            f"- {pos['ticker_symbol']}: {pos['quantity']} shares @ ${pos['trade_price']:.2f}"
        )
    return "\n".join(lines)


def _format_working_orders(working_orders: List[Dict[str, Any]]) -> str:
    if not working_orders:
        return "(No working orders.)"
    lines = []
    for order in working_orders:
        lines.append(
            f"- {order['ticker_symbol']} {order['side']} "
            f"{order['quantity']} @ ${order['price']:.2f} "
            f"({order.get('type', 'LIMIT')}, {order.get('duration', 'GOOD_TILL_CANCEL')})"
        )
    return "\n".join(lines)


def create_portfolio_manager(llm):
    def portfolio_manager_node(state) -> dict:
        symbols = state.get("symbols", [])
        ticker_decisions = state.get("ticker_decisions", [])
        portfolio = state.get("portfolio", [])
        working_orders = state.get("working_orders", [])
        account_balances = state.get("account_balances", {})
        trade_date = state.get("trade_date", "")

        cash = account_balances.get("cash", 0)
        net_liq = account_balances.get("net_liquidation", 0)

        prompt = f"""As the Portfolio Manager, synthesize today's per-ticker Asset Manager decisions into a portfolio-level trading plan for {trade_date}.

---

**Symbols in scope:** {', '.join(sorted(symbols))}

**Account Balances:**
- Cash: ${cash:,.2f}
- Net Liquidation: ${net_liq:,.2f}

**Current Portfolio:**
{_format_portfolio(portfolio)}

**Working Orders (pending):**
{_format_working_orders(working_orders)}

---

**Per-Ticker Analysis (final_trade_decision):**
{_format_ticker_decisions(ticker_decisions)}

---

**Your tasks:**

1. **Reconcile working orders** — for EACH existing working order, set one action:
   - **KEEP** if today's analysis still supports the pending order
   - **CANCEL** if analysis conflicts or the order should be withdrawn
   - **REPLACE** if analysis supports the direction but price/quantity should change (provide `replacement` order)

2. **New orders** — propose additional LIMIT / GOOD_TILL_CANCEL orders for symbols in scope based on ratings:
   - Buy / Overweight → BUY (integer shares)
   - Sell / Underweight → SELL (integer shares, only up to held quantity)
   - Hold → no new order unless rebalancing working orders

3. **Cash constraint** — total BUY notional for new orders plus any REPLACE buy orders must not exceed available cash after reconciling working orders (cash minus KEEP buy notional, plus freed cash from CANCEL).

4. **Order format** — every order must use type "LIMIT" and duration "GOOD_TILL_CANCEL"; quantities must be positive integers.

5. Only include tickers from the symbols-in-scope list.

6. **Output format** — respond ONLY by calling the PortfolioTradingPlan tool with populated `working_order_actions` (one per working order) and `orders` fields.{get_language_instruction()}"""

        plan = _invoke_portfolio_plan(llm, prompt)

        return {
            "structured_plan": plan,
            "portfolio_trading_plan": render_portfolio_trading_plan(plan),
        }

    return portfolio_manager_node
