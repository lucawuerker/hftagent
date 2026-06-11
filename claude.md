# QuantFundAgent

Master's thesis project (Mathematics & Finance, Imperial College London).

## Goal
Build LLM agents that operate like a professional quant fund: researching
features, designing strategies, and deploying them live.

## Architecture
A LangGraph multi-agent pipeline mirroring a quant fund's org chart:

- **Factor Researcher** – reads papers, brainstorms alpha ideas, generates
  factor code, backtests each one (recording its IC for reference) and keeps
  every factor that runs — IC magnitude is not a keep/drop gate.
- **Selector** – picks factors for a hypothesis.
- **Architect** – combines factors into a strategy via a refinement loop.
- **Statistician** – OOS tests, deflated Sharpe, accept/reject gate.
- **Portfolio Manager** – screens, allocates capital, monitors/retires
  strategies (single PM or a committee).

State flows through shared databases (factor / paper / strategy / portfolio).
The stage sequence lives in `quant_fund_agent/pipeline.py` so the notebook,
scripts, and backtests all drive the agents identically.

## Status
A working **MVP** of the full pipeline exists. Each stage and agent is
intended to be advanced significantly in future work.

**Pluggable data layer (Phases 0–4 done).** The fund no longer requires the
author's LOBSTER CSVs: any panel load goes through `quant_fund_agent/data/`
(provider abstraction + parquet cache + capability gating + frequency-aware
annualization). Providers: `lobster` (local CSVs), `yfinance` (key-free daily),
`fmp`, `alphavantage`. Run `python -m quant_fund_agent.setup` to pick a
provider/universe/timespan and write `quant.config.yaml`, or
`python -m quant_fund_agent.setup --assist "<plain-English description>"` to have
an LLM draft a config you confirm (`setup_assist.py`; validated to legal values,
falls back to the deterministic wizard with no LLM key). See
`docs/data-layer/ROADMAP.md`. Remaining: Phase 6 (multi-asset crypto/FX).

## Roadmap
- Longer backtests: simulate weekly researcher updates and trade over an
  extended period (the `pipeline.py` stages are built to be called on a
  schedule).
- Deepen the sophistication of every agent.

## Conventions
- Use Anthropic's MCP for agent/tool interactions.

## Python Environment
Always use the local virtual environment interpreter directly.
- Run python scripts: `./venv/bin/python path/to/script.py`
- Run tests: `./venv/bin/pytest`
- Install packages: `./venv/bin/pip install <package>`

Always update the global README.md and claude.md files after  significant changes to the last statusses mentioned in there. 
