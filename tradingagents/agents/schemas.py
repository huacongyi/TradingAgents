"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Asset Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class AssetRating(str, Enum):
    """5-tier rating used by the Research Manager and Asset Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Asset Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: AssetRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: Optional[float] = Field(
        default=None,
        description=(
            "Optional entry price target in the instrument's quote currency. "
            "Must be a single numeric value (for example 305.0), never a range "
            "(not '300-310') or free text. Use null when no specific level is available."
        ),
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description=(
            "Optional stop-loss price in the instrument's quote currency. "
            "Must be a single numeric value, never a range or free text. "
            "Use null when no level is set."
        ),
    )
    position_sizing: Optional[str] = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% allocation'.",
    )


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Asset Manager
# ---------------------------------------------------------------------------


class AssetDecision(BaseModel):
    """Structured output produced by the Asset Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: AssetRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: Optional[float] = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )


def render_asset_decision(decision: AssetDecision) -> str:
    """Render an AssetDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioOrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class PortfolioOrder(BaseModel):
    """A single LIMIT order with integer share quantity."""

    ticker_symbol: str = Field(description="Ticker symbol, e.g. NVDA or AAPL.")
    side: PortfolioOrderSide = Field(description="BUY or SELL.")
    quantity: int = Field(ge=1, description="Exact whole shares to trade.")
    price: float = Field(gt=0, description="LIMIT price in the instrument's quote currency.")
    type: Literal["LIMIT"] = Field(default="LIMIT", description="Order type; always LIMIT.")
    duration: Literal["GOOD_TILL_CANCEL"] = Field(
        default="GOOD_TILL_CANCEL",
        description="Order duration; always GOOD_TILL_CANCEL.",
    )


class WorkingOrderAction(BaseModel):
    """Action to take on an existing working order."""

    ticker_symbol: str = Field(description="Ticker of the working order to act on.")
    action: Literal["KEEP", "CANCEL", "REPLACE"] = Field(
        description=(
            "KEEP to leave the working order unchanged, CANCEL to remove it, "
            "or REPLACE to cancel and substitute with replacement."
        ),
    )
    replacement: Optional[PortfolioOrder] = Field(
        default=None,
        description="Required when action is REPLACE; the new order to submit.",
    )


class PortfolioTradingPlan(BaseModel):
    """Portfolio-level trading plan produced by the Portfolio Manager."""

    executive_summary: str = Field(
        description=(
            "Concise summary of today's portfolio actions, cash usage, "
            "and rationale for working-order changes. Two to four sentences."
        ),
    )
    working_order_actions: list[WorkingOrderAction] = Field(
        default_factory=list,
        description=(
            "One entry per existing working order. Reconcile each against "
            "today's per-ticker analysis (KEEP / CANCEL / REPLACE)."
        ),
    )
    orders: list[PortfolioOrder] = Field(
        default_factory=list,
        description=(
            "New orders to place today (excluding replacements, which belong "
            "in working_order_actions). Integer shares only; LIMIT / GTC."
        ),
    )


def _render_order(order: PortfolioOrder) -> str:
    return (
        f"- **{order.ticker_symbol}** {order.side.value} "
        f"{order.quantity} @ ${order.price:.2f} ({order.type}, {order.duration})"
    )


def render_portfolio_trading_plan(plan: PortfolioTradingPlan) -> str:
    """Render a PortfolioTradingPlan to markdown for display and logging."""
    parts = [
        f"**Executive Summary**: {plan.executive_summary}",
        "",
        "**Working Order Actions**:",
    ]
    if plan.working_order_actions:
        for action in plan.working_order_actions:
            line = f"- **{action.ticker_symbol}**: {action.action}"
            if action.action == "REPLACE" and action.replacement:
                line += f" → {_render_order(action.replacement).lstrip('- ')}"
            parts.append(line)
    else:
        parts.append("- (none)")

    parts.extend(["", "**New Orders**:"])
    if plan.orders:
        for order in plan.orders:
            parts.append(_render_order(order))
    else:
        parts.append("- (none)")

    return "\n".join(parts)
