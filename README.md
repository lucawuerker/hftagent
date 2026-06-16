# QuantFundAgent

Most of this detailed README is AI-generated. A short overview of the different folders and interesting files:

The core logic and all agents sit within the quant_fund_agent folder. The databases (the factors, papers and strategies) sit within the data folder. The diagrams folder contains diagrams of the system architecture as well as the architecutre of every single agents as both .png and .excalidraw files. prelim_files and tests contain tests or other files that have been created earlier and are not relevant to the current state of the project, but might still be of future interest for the thesis. 

The notebook demo_pipeline.ipynb demonstrates the different agents and there outputs. I preran all cells with my API keys and the data I downloaded. Rerunning them will not work, since it requires API keys as well as the data, which I can not push to Github because the files are too large. 

run_fund.py is the analogue part to demp_pipeline.ipynb that runs the whole pipeline as a python script with the same caveats regarding LLM and data availability.

Master_Thesis_Outline.pdf is a pdf document that describes the current stage of the project as well as further directions and things to work on. 



--------------- (mainly) AI generated README --------------------------


A LangGraph multi-agent system that mirrors the workflow of a quantitative hedge fund: research alpha factors, design and validate strategies, and allocate capital across a growing book of strategies.

Data are **10-second US large-cap equity bars** (LOBSTER-style CSVs). Agents reason with LLMs; backtests, model fitting, statistical tests, and portfolio maths run through deterministic Python backends exposed as [MCP](https://modelcontextprotocol.io) servers.

Master's thesis project — Mathematics & Finance, Imperial College London.

## Features

- **Five specialised agents** — Factor Researcher, Selector, Architect, Statistician, Portfolio Manager — wired as LangGraph subgraphs with shared JSON databases.
- **Factor-as-code** — seed alphas in version-controlled Python; the Researcher can generate new `BaseFactor` subclasses from papers.
- **ML strategy design** — the Architect picks from a catalog of linear models, tree ensembles, and gradient boosters (plus a static-weights baseline), iterates in a refinement loop, and persists fitted models for out-of-sample testing.
- **Statistical gate** — deflated Sharpe ratio and held-out OOS backtest before any strategy enters the book.
- **Portfolio construction** — seven allocation methods (equal weight through hierarchical risk parity), three risk personalities, optional multi-PM committee.
- **MCP toolboxes** — heavy compute (panel loads, fits, IC tests) runs in persistent stdio servers with an in-process fallback for tests and debugging.
- **End-to-end scripts** — one command runs research → strategy pipeline → persist → PM rebalance (`run_fund.py`).
- **Walk-forward backtest** — `run_backtest.py` runs the fund *through time*: weekly research/strategy/PM meetings, frozen strategies, a consolidated **netted** (or independent-pod) book, and spread-aware costs — decoupled from the agents so the same code runs live.

## Architecture

```
                     ┌──────────────────────┐
                     │     Orchestrator      │
                     │  (routes to agents)   │
                     └─┬───┬─────┬─────┬───┬─┘
                       │   │     │     │   │
       ┌───────────────┘   │     │     │   └────────────────┐
       ▼                   ▼     ▼     ▼                    ▼
 ┌──────────────┐ ┌──────────────┐ ┌─────────┐ ┌──────────────┐ ┌────────────────┐
 │ Factor       │ │ Selector     │ │Architect│ │Statistician  │ │ Portfolio Mgr  │
 │ Researcher   │ │              │ │         │ │              │ │ (+ committee)  │
 └──────┬───────┘ └──────┬───────┘ └────┬────┘ └──────┬───────┘ └────────┬───────┘
        │                │              │             │                  │
        └────────────────┴──────────────┴─────────────┴──────────────────┘
                                        │
                    Shared databases (JSON + per-strategy return CSVs)
                    factors · papers · strategies · portfolios
```

The canonical stage order lives in `quant_fund_agent/pipeline.py` and is shared by `run_fund.py`, the orchestrator, and notebooks:

| Stage | Function | What happens |
|-------|----------|--------------|
| 1 | `run_research_session` | Optional: read papers, generate factors, IC-backtest, persist survivors |
| 2 | `run_strategy_pipeline` | Selector → Architect → Statistician |
| 3 | `persist_approved_strategy` | Register accepted strategy + PnL series in the strategy DB |
| 4 | `run_pm_rebalance` | Screen, allocate, monitor; write a `PortfolioRecord` |

The statistician *judges* strategies; step 3 is what populates the strategy book for the PM. Without it, the Portfolio Manager has nothing to allocate.

## Quick start

```bash
python -m venv venv && source venv/bin/activate   # or .venv
pip install -r requirements.txt

# API key for LangChain OpenAI (all agents default to gpt-4o-mini)
export OPENAI_API_KEY=sk-...

# 1. Backtest seed factors and build factor_db.json (requires ticker_data/)
./venv/bin/python run_all_factors.py

# 2. Full fund: build N strategies, then PM committee rebalance
./venv/bin/python run_fund.py --n-strategies 3

# Optional: include a factor-research session first
./venv/bin/python run_fund.py --research --n-strategies 3

# 3. Walk-forward backtest: weekly research/strategy/PM meetings, traded over time
./venv/bin/python run_backtest.py --start 2019-01-02 --end 2019-03-02 --n-tickers 8
```

Individual stages:

```bash
./venv/bin/python run_factor_research.py          # Factor Researcher only
./venv/bin/python run_pipeline.py                 # Selector → Architect → Statistician (no persist)
./venv/bin/python run_portfolio_manager.py        # PM over existing strategy book
./venv/bin/python -m quant_fund_agent.main        # orchestrator (single task)
```

### Comparing research LLMs

Mine factors with different research models into named **preruns**, then compare
their factor sets on three axes — single-factor IC, brute-force ML (catalog +
ensemble), and the full downstream agentic fund:

```bash
# 1. Mine ~50 factors per model into a self-contained prerun (dedup per-prerun
#    so each model's brainstorm isn't anchored by the others — fair A/B).
./venv/bin/python run_factor_research.py --name gpt4omini --model gpt-4o-mini \
  --dedup-scope prerun --target-factors 50
./venv/bin/python run_factor_research.py --name claude --model claude-3-5-sonnet-latest \
  --llm-provider anthropic --dedup-scope prerun --target-factors 50

# 2. Compare them → figures + tables + report.md + comparison.ipynb under
#    data/comparisons/<id>/.  IC + brute-force are LLM-free; --no-downstream
#    skips the only LLM-spending track for a fully offline run.
./venv/bin/python run_model_comparison.py --preruns gpt4omini,claude
QF_USE_MCP=0 ./venv/bin/python run_model_comparison.py --all --no-downstream
```

Factors that need data not yet downloaded (e.g. fundamentals) are filtered and
reported, so the comparison runs on the current data and re-runs unchanged once
the full LOBSTER universe / FMP membership is in place.  `run_model_comparison.py
--research --prerun-spec name=model[:provider] …` can also mine the preruns first.

### Workspaces & books (modularisation by config + prerun)

Researched factors **and** the strategies built from them are strictly isolated
per **(data config, research-LLM prerun)** under
`data/workspaces/<config>/preruns/<prerun>/` — its own factor DB, strategies,
return series, fitted `.joblib` artifacts, portfolio and showcase. The canonical
*main* factor library `data/factors/factor_db.json` holds **only the
seed/formulaic alphas** and is never written by research. Every factor/strategy is
stamped with a `provenance` (data-config hash + scope).

```bash
# Strip the legacy mixed main DB → seed-only main + a preserved `legacy` scope
# (backed up, idempotent; --dry-run to preview).  Run once.
./venv/bin/python scripts/migrate_main_seed_only.py --dry-run
./venv/bin/python scripts/migrate_main_seed_only.py --migrate-preruns

# Research + build a fund under a named scope (config derived from quant.config.yaml).
./venv/bin/python run_fund.py --research --prerun gpt4omini --n-strategies 3

# Compose a separate ACTIVE BOOK by pooling chosen scopes' factors + strategies
# for the PM (main is never mutated; cross-config pooling is warned, not blocked).
./venv/bin/python run_merge.py --from-config yfinance_equity_demo --into demo_book --run-pm
```

See `fund_showcase.ipynb` for an interactive walkthrough.

## Agents

### Factor Researcher

Reads academic PDFs, proposes factor ideas, generates Python implementations, and keeps every candidate that backtests successfully. The IC backtest records each factor's IC for reference but does not gate on it — a low-IC factor (e.g. volatility) can still be a useful feature in combination, so whether to *use* a factor is left to the downstream agents.

**Graph:** `agents/factor_research/graph.py` — `load_papers` → `brainstorm` → `generate_code` → `backtest_factors` → `filter_and_persist`

**Codegen:** `agents/factor_research/codegen.py` validates generated code (AST checks, import allow-list, smoke test on synthetic data) before registering factors.

**Seed vs researcher factors:** seed alphas live under `quant_fund_agent/factors/` (version-controlled). Researcher output goes to `factors/researcher/` (gitignored) and is tagged `source=RESEARCHER` in the factor DB. `purge_researcher_factors()` clears researcher state between simulation runs.

**Lookahead:** paper selection respects `published_date`; the IC backtest panel can be sliced to a `cutoff_date`.

### Selector

Loads the factor catalog, asks an LLM to formulate a trading hypothesis, then selects factors that best support it.

**Graph:** `agents/selector/graph.py`

### Architect

Turns the Selector's hypothesis into a tradable strategy in a **refinement loop**:

1. **Design** — pick model type, hyperparameters, position settings from the modeling catalog.
2. **Fit & backtest** — fit on an in-sample window (ML models use an internal train/valid split inside IS); metrics are never measured on fitting data.
3. **Evaluate** — approve or revise based on metrics, fit diagnostics, and trial history.

**Graph:** `agents/architect/graph.py`

**Model catalog** (`modeling/catalog.py` — menu exposed to the LLM via MCP `list_models`):

| Family | Models |
|--------|--------|
| Baseline | `static_weights` — fixed factor weights, no fitting |
| Linear | `linear_regression`, `ridge`, `lasso`, `elastic_net` |
| Tree | `random_forest`, `gradient_boosting` (sklearn) |
| Boosting | `xgboost`, `lightgbm` (optional; omitted from menu if not installed) |

Hyperparameters are validated and clamped server-side. Fitted ML models are saved as joblib artifacts under `data/strategies/models/` and reloaded for OOS testing (`strategies/model_strategy.py`) — never refit on held-out data.

The out-of-sample slice is reserved for the Statistician; the Architect never sees it. Every trial is recorded for the deflated Sharpe ratio (multiple-testing correction).

### Statistician

Runs registered statistical tests and produces an accept/reject verdict.

**Graph:** `agents/statistician/graph.py`

**Mandatory tests** (`statistics/tests/`):

- **Deflated Sharpe ratio** — Bailey & López de Prado; uses full Architect trial history.
- **Out-of-sample backtest** — reloads persisted model artifact, evaluates on held-out slice.

Optional tests can be added via the registry without changing the graph. Any mandatory `FAIL` forces rejection even if the LLM says accept.

### Portfolio Manager

Screens the strategy universe, constructs portfolio weights, monitors live performance, and updates deployment status.

**Graph:** `agents/portfolio_manager/graph.py` — load universe → monitor → screen → construct → finalise

**Modes:** `SELECTOR` (rule-based, reproducible) or `ACTIVE` (LLM-assisted with rule fallbacks).

**Personalities** (`portfolio/personalities.py`): `defensive`, `balanced`, `aggressive` — each bundles screening filters, target book size, and a default construction method.

**Construction methods** (`portfolio/construction.py`):

| Method | Description |
|--------|-------------|
| `equal_weight` | 1/N benchmark |
| `inverse_volatility` | Risk balancing without full covariance |
| `min_variance` | Markowitz minimum variance |
| `mean_variance` | Markowitz with risk aversion λ |
| `max_sharpe` | Tangency portfolio (Schaefer QP for long-only) |
| `risk_parity` | Equal risk contribution |
| `hierarchical_risk_parity` | Lopez de Prado HRP |
| `custom_llm` | ACTIVE mode: LLM supplies weights directly |

**Committee:** `agents/portfolio_manager/committee.py` runs multiple PMs in propose-only mode and aggregates into one consensus allocation (simple average, weighted average, or LLM moderator).

**Strategy DB:** `StrategyDatabase` caches correlation/covariance matrices and supports incremental `append_returns()` for simulation loops.

## MCP tooling

LLM steps stay as plain `llm.invoke()` inside each agent. Deterministic work — panel loads, model fitting, IC backtests, statistical tests, portfolio optimisation — runs behind MCP stdio servers so heavy data stays server-side and only JSON crosses the boundary.

| Server | Agent | Tools |
|--------|-------|-------|
| `mcp/modeling_server.py` | Architect | `list_models`, `fit_and_backtest` |
| `mcp/catalog_server.py` | Selector | `load_factor_catalog` |
| `mcp/research_server.py` | Factor Researcher | `load_papers`, `existing_factor_ids`, `materialise_factor`, `backtest_factors`, `persist_results` |
| `mcp/statistics_server.py` | Statistician | `list_tests`, `run_tests` |
| `mcp/portfolio_server.py` | Portfolio Manager | `screen_strategies`, `construct_portfolio`, `expected_portfolio_metrics` |

Each server wraps functions in `mcp/*_service.py`. Agents call `mcp/*_client.py`, which uses a shared synchronous bridge (`mcp/_bridge.py`) — one persistent subprocess per server on a background event loop.

**Disable MCP** (in-process fallback, identical results — used in tests):

```bash
export QF_USE_MCP=0
# or per-server: MODELING_USE_MCP=0, RESEARCH_USE_MCP=0, etc.
```

Database mutations (flag/retire strategies, save portfolios) stay in the agent graphs because they use live in-memory DB handles shared with the committee.

## Backtesting (walk-forward)

`run_fund.py` is a single research → strategy → PM pass. `run_backtest.py` runs
the fund **through time**: the `quant_fund_agent/simulation/` harness steps over a
date range on a weekly grid, holding **research / strategy / PM meetings** on a
schedule and trading the resulting book on genuinely unseen data with realistic
execution.

The harness is **deliberately separate from the agents** — it drives them only
through the `pipeline.py` seam and is never imported by an agent, so the same
agent code path runs in a backtest and in live trading.

```bash
# Two months, weekly meetings, consolidated netted book (default).
./venv/bin/python run_backtest.py --start 2019-01-02 --end 2019-03-02 --n-tickers 8

# Compare the independent-pod execution model on the same span.
./venv/bin/python run_backtest.py --start 2019-01-02 --end 2019-03-02 --execution pod

# Smoke run: shrink the warm-up so meetings fire on a short span.
QF_USE_MCP=0 ./venv/bin/python run_backtest.py --start 2019-01-02 --end 2019-02-15 \
  --warmup 2W --initial-strategies 2 --n-strategies 1
```

**How it works** (all knobs live in `simulation/config.py::BacktestConfig`):

- **Warm-up** — no meetings or trading until `warmup` (default 1 month) of history
  exists, so the agents always have enough data to research and fit on.
- **Cadence** — `research_every` / `strategy_every` / `pm_rebalance_every` (default
  weekly). The first strategy meeting builds `initial_strategies` (bootstrap), every
  meeting after builds `n_strategies_per_meeting`.
- **No look-ahead** — at each weekly *cutoff* the Architect/Statistician only see
  data strictly *before* it (threaded as `cutoff_date`); the live trading window is
  strictly *after* it. Strategies are **frozen on approval** (artifacts never refit);
  the PM monitors live PnL and may retire them; only new research adds strategies.

**Trade execution** — per-strategy positions become the traded book one of two ways
(`--execution`, default `netted`):

| Model | Same/opposite signals on a ticker | `max_positions` | Costs |
|-------|-----------------------------------|-----------------|-------|
| `netted` (default) | summed into **one fund book** per ticker — opposite signals **net out**, agreeing ones reinforce | **fund-level** cap on the consolidated book (+ per-name cap, gross-leverage cap) | charged once on the **net** turnover |
| `pod` | each strategy trades its own book — **no netting** | per-strategy | each strategy pays its own |

Costs are **spread-aware** (½ the quoted `effSpread` from the panel) **+ a fixed
commission** (`--commission-bps`); there is no market-impact term.

**Outputs** land in `data/backtests/<run_id>/`: `equity.csv` (fund return + drawdown),
`fund_metrics.json` (Sharpe / Sortino / Calmar / maxDD / cost / turnover),
`attribution.csv` (per-strategy contribution), `meetings.jsonl`, and the run's
`config.json` + strategy/portfolio DBs (scoped to the run, so the live book is never
clobbered). `fund_showcase.ipynb` §11 renders the latest run.

## Data

**You don't need the author's LOBSTER data to run this.** Data loads through a
pluggable provider layer (`quant_fund_agent/data/`); pick one with the guided
wizard:

```bash
cp .env.example .env                  # add OPENAI_API_KEY (+ FMP/AV keys if used)
python -m quant_fund_agent.setup      # choose provider/universe/timespan → quant.config.yaml
./venv/bin/python run_fund.py --n-strategies 1
```

Prefer plain English? `--assist` lets an LLM draft the config, which the wizard
then shows for you to confirm:

```bash
python -m quant_fund_agent.setup --assist "US tech mega-caps, last 2 years, daily"
```

The proposal is only ever a *suggested default* (explicit flags still win, every
value is validated against real providers/presets/dates), and the wizard works
without an LLM key if you skip `--assist`.

| Provider | Key | Notes |
|----------|-----|-------|
| `yfinance` | none | Daily OHLCV, adjusted. Easiest clone-and-run. |
| `fmp` | `FMP_API_KEY` | FMP stable API, split/div-adjusted daily. |
| `alphavantage` | `ALPHAVANTAGE_API_KEY` | Daily; free tier is unadjusted + ~100 bars + ~25 req/day. |
| `lobster` | none | The original 10-second microstructure CSVs under `ticker_data/`. |

**Multi-asset:** all three API vendors serve **equity, crypto and FX**. Pick
`asset_class` in the wizard; crypto/FX use canonical `BASE-QUOTE` symbols
(`BTC-USD`, `EUR-USD`) with `crypto_demo` / `fx_demo` presets. Annualization is
**calendar-aware** — crypto (7-day trading) annualizes at 365, equities/FX at 252,
inferred from the data itself.

Vendor pulls are parquet-cached under `data/market/`. Factors are **capability-gated**
to the active provider's fields (microstructure factors hide on plain-OHLCV
vendors), and annualization is inferred from the data's frequency.

**Beyond OHLCV — fundamentals, estimates & events.** On an equity universe, FMP
and AlphaVantage also supply non-price fields — `sector`, `peRatio`, `roe`,
`revenue`, `eps`, analyst `epsEstimate`/`revenueEstimate`, and earnings
`epsSurprise`. These are **point-in-time**: each value is stamped at its filing /
report date (or fiscal-end + a reporting lag) and forward-filled onto the daily
panel, so a backtest never sees a number before it was public. A factor just
declares the field in `inputs` and the gating layer routes it to a capable
provider (`QF_FUNDAMENTALS=0` to opt out). See
[`FUNDAMENTAL_AND_ALT_DATA.md`](docs/data-layer/FUNDAMENTAL_AND_ALT_DATA.md);
sentiment + macro are the next (planned) fields.

**LOBSTER specifics:** place CSVs under `ticker_data/` (or set `DATA_DIR`). The
loader (`backtesting/data_loader.py`) builds an aligned panel of OHLCV plus
microstructure fields (`orderFlow`, `lobImb`, `spread`, `nbTrades`, etc.) on a
shared 10-second index.

Persisted state (modularised by config + prerun):

```
data/
├── factors/factor_db.json               # MAIN: seed/formulaic alphas ONLY (never written by research)
├── papers/index.json + pdfs/            # Paper metadata and PDFs
├── workspaces/<config>/                 # one data config (e.g. yfinance_equity_demo)
│   ├── config.snapshot.json
│   └── preruns/<prerun>/                # one research-LLM batch
│       ├── factors/factor_db.json       # this scope's RESEARCHER factors
│       ├── strategies/{strategy_db.json, returns/*.csv, models/*.joblib}
│       ├── portfolio/portfolio_db.json
│       └── showcase.json
└── books/<name>/                        # composed active book (run_merge.py output) for the PM
    └── {factors, strategies, portfolio}/…
```

## Project structure

```
QuantFundAgent/
├── quant_fund_agent/
│   ├── pipeline.py              # Stage functions (research / strategy / persist / PM)
│   ├── workspace.py              # Single source of truth for (config, prerun) layout: Scope / Book
│   ├── merge.py                  # Compose an active book by pooling scopes (factors + strategies)
│   ├── orchestrator.py            # Top-level task router
│   ├── schemas.py                 # Pydantic records
│   ├── agents/                    # LangGraph subgraphs (one dir per agent)
│   ├── backtesting/               # Panel loader, IC engine, strategy backtester
│   ├── databases/                 # Factor, paper, strategy, portfolio DBs
│   ├── factors/                   # Seed + researcher factor implementations
│   ├── modeling/                  # Model catalog, features, train/fit pipeline
│   ├── mcp/                       # MCP servers, clients, shared services
│   ├── portfolio/                 # Construction methods, personalities, correlations
│   ├── statistics/                # Test registry + implementations
│   ├── strategies/                # BaseStrategy, ModelStrategy, DynamicStrategy
│   └── simulation/                # Walk-forward backtest harness (separate from agents)
├── run_fund.py                    # End-to-end fund demo (single pass, per scope)
├── run_merge.py                   # Pool scopes into an active book for the PM
├── run_backtest.py                # Walk-forward backtest (weekly meetings over time)
├── run_pipeline.py                # Strategy pipeline only
├── run_factor_research.py
├── run_portfolio_manager.py
├── run_all_factors.py
├── scripts/migrate_main_seed_only.py  # One-shot: make main factor DB seed-only
├── fund_showcase.ipynb
└── tests/                         # Including test_workspace/merge/migration/provenance.py
```

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | Required for LLM calls |
| `DATA_DIR` | `ticker_data` | Panel data directory |
| `ARCHITECT_N_TICKERS` | all | Cap universe size (memory); e.g. `10` |
| `QF_USE_MCP` | `1` | Global MCP toggle (`0` = in-process) |
| `FACTOR_RESEARCH_LLM_MODEL` | `gpt-4o-mini` | Per-agent model overrides |
| `SELECTOR_LLM_MODEL` | `gpt-4o-mini` | |
| `ARCHITECT_LLM_MODEL` | `gpt-4o-mini` | |
| `STATISTICIAN_LLM_MODEL` | `gpt-4o-mini` | |
| `PM_LLM_MODEL` | `gpt-4o-mini` | |

On macOS, XGBoost/LightGBM need OpenMP (`brew install libomp`). If unavailable, those models are omitted from the Architect menu automatically.

## Development

```bash
./venv/bin/pytest                    # full suite (set PYTHONPATH=. if needed)
./venv/bin/pytest tests/test_mcp_modeling.py -q
```

Core design choices:

- **Factors and strategies as code** — computation lives in Python classes; metadata and backtest results live in JSON records linked by ID.
- **Registry pattern** — `@register_factor`, `@register_test`; auto-discovery at startup.
- **IS / valid / OOS discipline** — Architect fits on IS-train, reports on IS-valid; Statistician evaluates on OOS with a frozen artifact.
- **Incremental simulation APIs** — `StrategyDatabase.append_returns`, weekly `pipeline.py` stages, PM correlation cache — built for a scheduled multi-week backtest loop.

## License

Thesis / academic use. Add a license file if you open-source the repository publicly.
