from tradingagents.graph.trading_graph import TradingAgentsGraph
# from tradingagents.default_config import DEFAULT_CONFIG # -: Hua Congyi
# from tradingagents.gemini_config import DEFAULT_CONFIG # +: Hua Congyi
# from tradingagents.minimax_nvidia_config import DEFAULT_CONFIG # +: Hua Congyi
from tradingagents.minimax_cn_config import DEFAULT_CONFIG # +: Hua Congyi

# DEFAULT_CONFIG already applies TRADINGAGENTS_* env-var overrides
# (llm_provider, deep_think_llm, quick_think_llm, backend_url, etc.),
# so users can switch models or endpoints purely via .env without
# editing this script. Override individual keys here only when you
# want a hard-coded value that should ignore the environment.
config = DEFAULT_CONFIG.copy()

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
# _, decision = ta.propagate("NVDA", "2024-05-22") # -: Hua Congyi
_, decision = ta.propagate("NVDA", "2026-06-09") # +: Hua Congyi
print(decision)
ta.print_node_timing_report()

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
