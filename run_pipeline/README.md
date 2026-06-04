# Full pipeline runner (real market data)

These scripts run the quant-fund agent pipeline **end-to-end on the local
order-book / price panel** (`ticker_data/`) and write to the **real** databases
(`data/factors`, `data/strategies`, `data/portfolio`). Every market-data step
actually runs: the Factor Researcher's IC backtest, the Architect's fit +
backtest refinement loop, the Statistician's out-of-sample tests, and the
Portfolio Manager's allocation.

This is the up-to-date counterpart to the two other example sets:

* `prelim_files/` — older standalone scripts that predate
  `quant_fund_agent/pipeline.py`; these replace them and drive the agents
  through that reusable seam instead of re-implementing orchestration.
* `showcase_pipeline/` — clean per-step scripts that deliberately **stop before**
  any market-data step and write to *sandbox* DB copies (so the public repo
  needs no proprietary data). These go all the way through, on real data.

| Stage | Script | Runs |
|------|--------|------|
| 1 · Factor Researcher | `1_factor_research.py` | papers → ideas → `BaseFactor` code → **IC backtest** → keep every factor that runs (IC recorded, not gated) |
| 2 · Selector | `2_selector.py` | factor catalog → hypothesis → selected factors |
| 3 · Architect | `3_architect.py` | design → **fit + backtest** refinement loop → accept/revise |
| 4 · Statistician | `4_statistician.py` | Selector → Architect → **OOS accept/reject gate** → persist accepted strategy |
| 5 · Portfolio Manager | `5_portfolio_manager.py` | screen → allocate (single PM or committee) over the book |
| — · Whole pipeline | `run_pipeline.py` | all of the above, with `--skip-*` flags |

## Prerequisites

* The project virtual environment.
* The local market-data panel at `ticker_data/` (or pass `--data-dir`).
* An `OPENAI_API_KEY` in a `.env` file at the repository root.

## Running

From anywhere in the repo (the scripts `cd` to the project root themselves):

```bash
# Per-step (small universe keeps the demo fast):
./venv/bin/python run_pipeline/1_factor_research.py --n-papers 1 --n-ideas 2 --n-tickers 8
./venv/bin/python run_pipeline/2_selector.py
./venv/bin/python run_pipeline/3_architect.py --n-tickers 8 --max-iterations 2
./venv/bin/python run_pipeline/4_statistician.py --n-tickers 8 --max-iterations 2
./venv/bin/python run_pipeline/5_portfolio_manager.py --committee defensive,balanced,aggressive

# Whole pipeline (skip the slow research stage; 1 strategy; tiny universe):
./venv/bin/python run_pipeline/run_pipeline.py --skip-research --n-strategies 1 --n-tickers 8
```

The per-step scripts run in order for a coherent story (Stage 1's kept factors
become selectable in Stage 2, whose hypothesis feeds Stages 3–4, whose persisted
strategy feeds Stage 5), or you can run any one on its own — each is
self-contained. Note that **Stages 3 and 4 each re-run the Selector + Architect**
(the per-step scripts are intentionally standalone); `run_pipeline.py` does one
chained pass instead.

## Common flags

* `--n-tickers N` — universe cap for the market-data backtests (sets
  `ARCHITECT_N_TICKERS`). `ticker_data/` is several GB across 50+ tickers, so a
  small cap (e.g. `8`–`15`) keeps memory and runtime sane. `0` = full universe.
* `--data-dir PATH` — panel location (default `ticker_data`).
* `--mcp` — run agent tools over their MCP stdio servers (the production path).
  **By default the scripts run tools in-process** (`QF_USE_MCP=0`) — faster and
  with no subprocesses.
* `--fresh` (Stage 4 + `run_pipeline.py`) — start from an empty strategy/portfolio
  book instead of the saved one.

## What gets written

Unlike the showcase, these scripts use the **real** databases:

* generated factors → `data/factors/factor_db.json` (status `backtested`; every
  factor that runs is kept — IC is recorded but not used as a gate)
* accepted strategies → `data/strategies/strategy_db.json` (+ returns under
  `data/strategies/returns/`, ML artifacts under `data/strategies/models/`)
* portfolio allocations → `data/portfolio/portfolio_db.json`

Use `--fresh` (or check `git status` / restore the JSON files) if you want to
start clean.
