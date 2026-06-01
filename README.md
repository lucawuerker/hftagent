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
```

Individual stages:

```bash
./venv/bin/python run_factor_research.py          # Factor Researcher only
./venv/bin/python run_pipeline.py                 # Selector → Architect → Statistician (no persist)
./venv/bin/python run_portfolio_manager.py        # PM over existing strategy book
./venv/bin/python -m quant_fund_agent.main        # orchestrator (single task)
```

See `fund_showcase.ipynb` for an interactive walkthrough.

## Agents

### Factor Researcher

Reads academic PDFs, proposes factor ideas, generates Python implementations, and keeps candidates that pass an IC threshold.

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

## Data

Place LOBSTER CSVs under `ticker_data/` (or set `DATA_DIR`). The loader (`backtesting/data_loader.py`) builds an aligned panel of OHLCV plus microstructure fields (`orderFlow`, `lobImb`, `spread`, `nbTrades`, etc.) on a shared 10-second index.

Persisted state:

```
data/
├── factors/factor_db.json       # FactorRecord registry
├── papers/index.json + pdfs/  # Paper metadata and PDFs
├── strategies/
│   ├── strategy_db.json       # StrategyRecord registry
│   ├── returns/*.csv          # Per-strategy PnL series
│   └── models/*.joblib        # Fitted ML artifacts
└── portfolio/portfolio_db.json
```

## Project structure

```
QuantFundAgent/
├── quant_fund_agent/
│   ├── pipeline.py              # Stage functions (research / strategy / persist / PM)
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
│   └── strategies/                # BaseStrategy, ModelStrategy, DynamicStrategy
├── run_fund.py                    # End-to-end fund demo
├── run_pipeline.py                # Strategy pipeline only
├── run_factor_research.py
├── run_portfolio_manager.py
├── run_all_factors.py
├── fund_showcase.ipynb
└── tests/                         # Including test_mcp_*.py parity tests
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
