# TradingAgents (Study Fork)

Multi-agent LLM financial trading framework — a personal study fork tailored for experimenting with providers, configs, and the agent pipeline.

## Origin

This repository is forked from [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) at release **v0.2.5**. It is **not** an official continuation of the upstream project.

- Upstream: https://github.com/TauricResearch/TradingAgents
- Paper: [TradingAgents: Multi-Agents LLM Financial Trading Framework](https://arxiv.org/abs/2412.20138) (arXiv:2412.20138)
- License: [Apache 2.0](LICENSE)

## Purpose

This fork exists for **personal study**: trying different LLM backends (Gemini, MiniMax China, NVIDIA NIM), tuning configs, and learning how the multi-agent trading workflow behaves. Changes here are oriented toward local experiments, not upstream contribution.

Fork-specific changes are listed under `[Unreleased]` in [CHANGELOG.md](CHANGELOG.md).

## Study configs

Besides the upstream [`tradingagents/default_config.py`](tradingagents/default_config.py), this fork adds preset configs:

| Module | Provider | Use case |
|--------|----------|----------|
| [`tradingagents/gemini_config.py`](tradingagents/gemini_config.py) | Google (`gemini-3.1-flash-lite`) | Gemini API experiments |
| [`tradingagents/minimax_cn_config.py`](tradingagents/minimax_cn_config.py) | MiniMax China (`MiniMax-M2.7`) | Official China API (`MINIMAX_CN_API_KEY`) |
| [`tradingagents/minimax_nvidia_config.py`](tradingagents/minimax_nvidia_config.py) | OpenAI-compatible + NVIDIA NIM | MiniMax M2.7 via `integrate.api.nvidia.com` |

Switch configs in [`main.py`](main.py) by changing the `DEFAULT_CONFIG` import, or override via `TRADINGAGENTS_*` env vars (see upstream defaults).

**NVIDIA NIM:** set `TRADINGAGENTS_LLM_BACKEND_URL=https://integrate.api.nvidia.com/v1` and either `OPENAI_API_KEY` or `NVIDIA_API_KEY` (see [`.env.example`](.env.example)).

Smoke-test scripts live under [`scripts/`](scripts/) (`test_gemini_models_langgraph.py`, `test_minimax_models_langgraph.py`, `test_nvidia_minimax_models_langgraph.py`).

## Git remotes

```bash
git remote -v
# origin    https://github.com/huacongyi/TradingAgents.git   (this fork)
# upstream  https://github.com/TauricResearch/TradingAgents.git
```

To sync with upstream later: `git fetch upstream && git merge upstream/main`.

---

## TradingAgents Framework

TradingAgents is a multi-agent trading framework that mirrors the dynamics of real-world trading firms. Specialized LLM-powered agents — fundamental analysts, sentiment experts, technical analysts, researchers, trader, and risk/asset managers — collaboratively evaluate market conditions and inform trading decisions through structured debate.

<p align="center">
  <img src="assets/schema.png" style="width: 100%; height: auto;">
</p>

> This framework is for **research and education only**. Trading performance depends on model choice, temperature, data quality, and other non-deterministic factors. **Not financial, investment, or trading advice.**

### Analyst Team

- **Fundamentals Analyst** — company financials and intrinsic value
- **Sentiment Analyst** — news, StockTwits, and Reddit sentiment
- **News Analyst** — macro news and economic indicators
- **Technical Analyst** — indicators (MACD, RSI, etc.) and price patterns

### Researcher Team

Bullish and bearish researchers debate analyst findings to balance upside and risk.

### Trader Agent

Turns research into a concrete buy/sell/hold proposal with optional price levels.

### Risk Management and Asset Manager

Risk debaters stress-test the proposal; the Asset Manager approves or rejects the trade.

## Installation and CLI

### Installation

```bash
git clone https://github.com/huacongyi/TradingAgents.git
cd TradingAgents

conda create -n tradingagents python=3.13
conda activate tradingagents

pip install .
```

### Docker

```bash
cp .env.example .env  # add your API keys
docker compose run --rm tradingagents
```

For Ollama: `docker compose --profile ollama run --rm tradingagents-ollama`

### Required APIs

Set the key for your chosen provider:

```bash
export OPENAI_API_KEY=...
export NVIDIA_API_KEY=...          # optional; NVIDIA NIM when using OpenAI-compatible backend
export GOOGLE_API_KEY=...
export ANTHROPIC_API_KEY=...
export MINIMAX_CN_API_KEY=...      # MiniMax China
# ... see .env.example for full list
```

Or copy `.env.example` to `.env` and fill in keys.

### CLI Usage

```bash
tradingagents
python -m cli.main
```

## Python Usage

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

ta = TradingAgentsGraph(debug=True, config=DEFAULT_CONFIG.copy())
_, decision = ta.propagate("NVDA", "2026-01-15")
print(decision)
```

Study fork example (MiniMax China preset):

```python
from tradingagents.minimax_cn_config import DEFAULT_CONFIG
# ... same as above
```

Override models and debate rounds via config or `TRADINGAGENTS_*` env vars. See `tradingagents/default_config.py`.

### Node timing (study fork)

Each graph run records per-execution wall-clock time for every node (analysts, tool nodes, message-clear nodes, debators, managers). Call `print_node_timing_report()` after `propagate()` to print a timing table — useful for spotting slow LLM calls vs. data-fetch steps.

```python
_, decision = ta.propagate("NVDA", "2026-06-09")
ta.print_node_timing_report()
```

## Persistence and Recovery

- **Decision log** — append-only at `~/.tradingagents/memory/trading_memory.md` (`TRADINGAGENTS_MEMORY_LOG_PATH` to override)
- **Checkpoint resume** — opt-in via `--checkpoint`; state under `~/.tradingagents/cache/checkpoints/`

## Citation

If you use the original TradingAgents work, please cite:

```
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
      title={TradingAgents: Multi-Agents LLM Financial Trading Framework},
      author={Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
      year={2025},
      eprint={2412.20138},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2412.20138},
}
```
