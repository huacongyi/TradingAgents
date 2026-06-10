from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.graph.portfolio_trading_graph import PortfolioTradingGraph
from tradingagents.agents.schemas import render_portfolio_trading_plan
# from tradingagents.default_config import DEFAULT_CONFIG # -: Hua Congyi
# from tradingagents.gemini_config import DEFAULT_CONFIG # +: Hua Congyi
# from tradingagents.minimax_nvidia_config import DEFAULT_CONFIG # +: Hua Congyi
from tradingagents.minimax_cn_config import DEFAULT_CONFIG # +: Hua Congyi
from datetime import datetime
from tqdm import tqdm
import os
import json

portfolio = [{"ticker_symbol": "NVDA", "trade_price": 201.83, "quantity": 3.0}, {"ticker_symbol": "AAPL", "trade_price": 253.95, "quantity": 6.0}]
working_orders = [{"ticker_symbol": "NVDA", "side": "BUY", "quantity": 1.0, "price": 190.00, "type": "LIMIT", "duration": "GOOD_TILL_CANCEL"}, {"ticker_symbol": "AAPL", "side": "BUY", "quantity": 2.0, "price": 260.00, "type": "LIMIT", "duration": "GOOD_TILL_CANCEL"}]
watchlist = ["NASA", "TSLA"]
account_balances = {"net_liquidation": 10000, "cash": 7870.81}
symbols_to_trade = set([asset["ticker_symbol"] for asset in portfolio] + [asset["ticker_symbol"] for asset in working_orders] + watchlist)


# DEFAULT_CONFIG already applies TRADINGAGENTS_* env-var overrides
# (llm_provider, deep_think_llm, quick_think_llm, backend_url, etc.),
# so users can switch models or endpoints purely via .env without
# editing this script. Override individual keys here only when you
# want a hard-coded value that should ignore the environment.
config = DEFAULT_CONFIG.copy()

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
# today = datetime.now().strftime("%Y-%m-%d")
today = "2026-06-09"
# _, decision = ta.propagate("NVDA", "2024-05-22") # -: Hua Congyi
for symbol in tqdm(symbols_to_trade):
    symbol_log_path = os.path.join(config["results_dir"], symbol, "TradingAgentsStrategy_logs", f"full_states_log_{today}.json")
    if os.path.exists(symbol_log_path):
        continue
    _, decision = ta.propagate(symbol, today) # +: Hua Congyi
    print(decision)
    # ta.print_node_timing_report()

# Portfolio-level trading plan
ptg = PortfolioTradingGraph(debug=True, config=config)
_, portfolio_plan = ptg.propagate(
    trade_date=today,
    symbols=symbols_to_trade,
    portfolio=portfolio,
    working_orders=working_orders,
    account_balances=account_balances,
)
print(render_portfolio_trading_plan(portfolio_plan))

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
