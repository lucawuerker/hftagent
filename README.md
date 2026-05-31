# QuantFundAgent

A LangGraph-based multi-agent system that mirrors the structure of a quantitative fund / prop trading firm.

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
 │ Researcher   │ │ Agent        │ │ Agent   │ │ Agent        │ │ Agent          │
 └──────┬───────┘ └──────┬───────┘ └────┬────┘ └──────┬───────┘ └────────┬───────┘
        │                │              │             │                  │
        ▼                ▼              ▼             ▼                  ▼
 read papers→     pick seed +     combine factors→ OOS tests +     screen + monitor +
 brainstorm →     researcher      refine loop  →   deflated SR  →  allocate capital;
 codegen →        factors for a   backtest IS      → approve /     flag / retire
 IC backtest →    hypothesis                          reject       under-performers
 keep good ones
        │                │              │             │                  │
        ▼                ▼              ▼             ▼                  ▼
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │                              Shared Databases                                    │
 │ ┌──────────┐ ┌──────────┐ ┌─────────────────────────┐ ┌───────────────────────┐ │
 │ │ Factor   │ │ Paper    │ │ Strategy DB             │ │ Portfolio DB          │ │
 │ │  DB      │ │  DB      │ │ + per-strategy returns  │ │ (PM audit trail)      │ │
 │ │          │ │          │ │ + correlation matrix    │ │                       │ │
 │ └──────────┘ └──────────┘ └─────────────────────────┘ └───────────────────────┘ │
 └─────────────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
QuantFundAgent/
├── quant_fund_agent/
│   ├── schemas.py                  # Pydantic records: FactorRecord, Paper, StrategyRecord, …
│   ├── state.py                    # LangGraph state definitions (FundState + per-agent)
│   ├── orchestrator.py             # Top-level graph that routes to agent subgraphs
│   ├── main.py                     # Entry point
│   │
│   ├── databases/                  # Metadata registries (persist to JSON)
│   │   ├── factor_db.py            # FactorRecord registry + source/session filtering + purge
│   │   ├── paper_db.py             # Paper registry with auto-discovery + lookahead filtering
│   │   ├── strategy_db.py          # StrategyRecord + returns persistence + correlation matrix
│   │   └── portfolio_db.py         # PortfolioRecord audit-trail registry
│   │
│   ├── factors/                    # Factor implementations (Python classes)
│   │   ├── base.py                 # BaseFactor ABC — every factor inherits this
│   │   ├── registry.py             # @register_factor decorator + lookup functions
│   │   ├── _discover.py            # Auto-imports all factor modules at startup
│   │   ├── ops.py                  # Operator primitives (rank, ts_rank, decay_linear, …)
│   │   ├── momentum/, mean_reversion/, microstructure/, …   # SEED factors (version-controlled)
│   │   └── researcher/             # RESEARCHER factors — auto-generated, gitignored,
│   │                               # purged between simulation runs
│   │
│   ├── strategies/                 # Strategy implementations (Python classes)
│   │   ├── base.py                 # BaseStrategy ABC — every strategy inherits this
│   │   ├── registry.py             # @register_strategy decorator + lookup functions
│   │   └── implementations/
│   │
│   ├── agents/                     # LangGraph subgraphs
│   │   ├── factor_research/        # Factor Researcher Agent
│   │   │   ├── graph.py            # LangGraph nodes + wiring
│   │   │   ├── state.py            # FactorResearcherState, FactorIdea, FactorCandidate
│   │   │   ├── prompts.py          # Brainstorm + codegen prompts (data context separated)
│   │   │   └── codegen.py          # Validate / write / import / smoke-test generated code
│   │   ├── selector/graph.py       # Picks factors for a hypothesis
│   │   ├── architect/graph.py      # Combines factors into a strategy (refinement loop)
│   │   ├── statistician/graph.py   # OOS tests, deflated Sharpe, approval gate
│   │   └── portfolio_manager/      # NEW — Portfolio Manager Agent
│   │       ├── graph.py            # load → monitor → screen → construct → finalise
│   │       ├── state.py            # PortfolioManagerState (mode, personality, outputs, propose_only)
│   │       ├── prompts.py          # ACTIVE-mode LLM prompts (monitor / screen / construction)
│   │       └── committee.py        # NEW — run N PMs, aggregate into one consensus PortfolioRecord
│   │
│   ├── portfolio/                  # NEW — pure-Python portfolio library used by the PM agent
│   │   ├── construction.py         # equal_weight, inverse_vol, min_var, mean_var,
│   │   │                           # max_sharpe, risk_parity, HRP + dispatcher
│   │   ├── correlations.py         # strategy × strategy correlation / covariance
│   │   └── personalities.py        # DEFENSIVE / BALANCED / AGGRESSIVE PM profiles
│   │
│   ├── backtesting/
│   │   ├── data_loader.py          # LOBSTER CSV → OHLCV panel dict
│   │   ├── data_split.py           # Chronological IS/OOS split
│   │   ├── engine.py               # Single-factor IC / ICIR backtest
│   │   └── strategy_backtester.py  # Strategy-level backtest
│   │
│   ├── pipeline.py                 # NEW — reusable stage functions (research /
│   │                               # strategy pipeline / persist / PM rebalance)
│   │                               # shared by the script, notebook & backtest
│   │
│   └── utils/
│       └── pdf.py                  # NEW — robust PDF text extraction (pypdf, char-capped)
│
├── data/
│   ├── factors/
│   │   └── factor_db.json          # Persisted FactorRecord registry (seed + researcher)
│   ├── papers/
│   │   ├── index.json              # Paper metadata (loaded by PaperDatabase)
│   │   └── pdfs/                   # Actual PDF files
│   ├── strategies/
│   │   ├── strategy_db.json        # Persisted StrategyRecord registry (with pm_status, correlations)
│   │   └── returns/                # Per-strategy PnL series (CSV); one file per strategy_id
│   └── portfolio/
│       └── portfolio_db.json       # PM audit trail: every PortfolioRecord ever produced
│
├── run_fund.py                     # NEW — whole fund, one command: research →
│                                   # strategy pipeline → persist → PM rebalance
├── run_pipeline.py                 # Selector → Architect → Statistician (no persist)
├── run_factor_research.py          # Invoke a single Factor Researcher session
├── run_portfolio_manager.py        # NEW — invoke a PM agent over the current strategy book
├── run_all_factors.py              # Regenerate factor_db.json from registered SEED factors
├── requirements.txt
├── .env.example
└── README.md
```

## Key Design Decisions

### Factor & Strategy as Code

Every factor is a Python class that inherits from `BaseFactor` and implements a `calc(data)` method.  Strategies inherit from `BaseStrategy` and implement `calc(factor_signals, data)`.  This keeps computation in version-controlled code while metadata (backtest results, status, trading-idea links) lives in the database records (`FactorRecord` / `StrategyRecord` in `schemas.py`).

The `class_name` field on each database record links it back to the implementation class:

```
FactorRecord (schemas.py)          BaseFactor subclass (factors/)
┌──────────────────────┐           ┌──────────────────────────┐
│ id: "rsi_mr"         │──────────▶│ RSIMeanReversionSignal   │
│ class_name:          │  registry │   factor_id = "rsi_mr"   │
│   "RSIMeanReversion" │  lookup   │   def calc(self, data)   │
│ status: "approved"   │           └──────────────────────────┘
│ source: "seed"       │
│ backtest_metrics: …  │
└──────────────────────┘
```

### Factor Organisation

Factors live under `factors/<category>/`, one file per factor:

```
factors/
├── momentum/                       ← SEED, version-controlled
│   ├── three_soldiers.py
│   ├── alpha_007.py … alpha_095.py
├── mean_reversion/                 ← SEED
├── microstructure/                 ← SEED
├── statistical_arbitrage/          ← SEED
├── volatility/                     ← SEED (add new categories as directories)
└── researcher/                     ← RESEARCHER, auto-generated, gitignored
    ├── __init__.py                 (only file tracked in git)
    └── <session-generated>.py …
```

At startup call `discover_factors()` to auto-import every module (seed *and* researcher) and populate the global registry.

### Seed vs Researcher Factors  *(new)*

`FactorRecord` carries two extra fields:

| Field                  | Purpose                                                                 |
| ---------------------- | ----------------------------------------------------------------------- |
| `source`               | `seed` (hard-coded baseline) or `researcher` (invented by the agent).   |
| `research_session_id`  | The session in which a researcher factor was created (e.g. `2019-W03`). |
| `trading_idea`         | The *why* — the thesis the agent justified its idea with.               |
| `code_path`            | Filesystem path of the generated `.py` file (researcher factors only).  |

The database supports filtering and bulk purging:

```python
from quant_fund_agent.databases import FactorDatabase
from quant_fund_agent.schemas import FactorSource

db = FactorDatabase()
db.load_from_json("data/factors/factor_db.json")

seeds        = db.list_factors(source=FactorSource.SEED)         # baseline only
researcher   = db.list_factors(source=FactorSource.RESEARCHER)   # everything the agent added
this_session = db.list_factors(source=FactorSource.RESEARCHER,
                               research_session_id="2019-W03")

removed = db.purge_researcher_factors()                          # wipe everything
removed = db.purge_researcher_factors("2019-W03")                # wipe one session
```

**Rationale.**  The 1-year simulation must always boot from a deterministic, reproducible baseline (the seed ensemble).  Factors invented mid-simulation by the Factor Researcher Agent are scoped to that run; on the next run we want a clean slate.  The `source` tag + filesystem-purgeable code directory + `purge_researcher_factors()` give us exactly that property without forcing two separate databases or two separate runtime paths.

### Paper Storage & Lookahead-Bias Prevention

Papers are stored in `data/papers/`:

- **`index.json`** — array of `Paper` objects with a mandatory `published_date` field.
- **`pdfs/`** — the actual PDF files, referenced by `file_path`.

`PaperDatabase.list_papers_before(cutoff_date)` returns only papers published before the cutoff.  The Factor Researcher Agent uses this when invoked inside a 1-year simulation: in week `N` the cutoff is set to that week's start, so the agent can only "have read" papers that genuinely existed at that point.

`PaperDatabase.auto_discover_pdfs(pdf_dir)` scans the `pdfs/` directory and registers any new files with default metadata, so dropping a new PDF into the folder is enough to make it visible to the agent on the next run.

### Registry Pattern

```python
from quant_fund_agent.factors import discover_factors, instantiate_factor

discover_factors()                          # imports all factor modules (seed + researcher)
factor = instantiate_factor("rsi_mean_reversion")  # returns RSIMeanReversionSignal()
signal = factor.calc(data)
```

## Factor Researcher Agent  *(new)*

### What it does

The Factor Researcher Agent is the **idea-generation stage** that feeds new alpha hypotheses into the ensemble.  One *research session* does:

1. **`load_papers`** — Pick `n_papers` papers (lookahead-bias filtered if `cutoff_date` is set), extract up to `max_chars_per_paper` characters of text from each PDF.
2. **`brainstorm`** — One LLM call sees the paper text + the existing factor catalog, returns `n_factor_ideas` distinct factor specs (id, name, category, trading thesis, source paper ids).
3. **`generate_code`** — For each idea, one LLM call returns the complete Python source of a `BaseFactor` subclass.  Each file is statically validated (forbidden tokens, import allow-list, exactly one `BaseFactor` subclass with the right `factor_id`), then imported, then smoke-tested on a synthetic OHLCV panel.  Failures are dropped without aborting the session.
4. **`backtest_factors`** — Each surviving candidate goes through the standard single-factor IC backtest (`backtesting/engine.py`) on the panel sliced up to `cutoff_date`.
5. **`filter_and_persist`** — Keep candidates whose absolute IC at `ic_target_horizon` ≥ `ic_threshold`; tag survivors with `source=RESEARCHER`, `research_session_id=<session>`, persist to the factor DB.  Reject the rest — both from the registry and from the filesystem — so nothing stale lingers.

### Why this design

- **Same execution path as seed factors.**  Generated `.py` files live inside `quant_fund_agent/factors/researcher/` and are picked up by the *existing* `discover_factors()`.  There is one and only one way to run a factor.  This was preferred over a "formula-string interpreter" or eval-based approach because (a) it preserves the "Factor as Code" invariant and (b) every researcher factor remains debuggable and reviewable as plain Python.
- **Source tag + filesystem segregation, not a second database.**  A second DB would have meant duplicating loading/saving/filtering logic everywhere; instead the `source` enum + `purge_researcher_factors` give us bulk-clear semantics in one place.  The directory split (`researcher/` is gitignored, seed factor directories are not) gives us clean reproducibility.
- **Generation runs aren't all-or-nothing.**  Every per-candidate failure (bad codegen, calc raises, IC too low) is logged and skipped.  A session that produces 1 good factor out of 3 ideas is still a useful session.
- **Static guardrails on generated code.**  The LLM is constrained to a small allow-list of imports (`pandas`, `numpy`, the factor base + ops modules).  A forbidden-token check (`eval`, `exec`, `os.system`, `subprocess`, network, filesystem) catches obvious red flags.  AST parsing enforces that the class shape matches the spec.  This is *not* a security sandbox — the LLM is treated as a cooperative coworker — but it dramatically reduces accidental damage.
- **IC gate at the target horizon.**  We use the existing IC backtest pipeline (so seed and researcher factors are evaluated identically) and apply `|IC@h| ≥ ic_threshold` at a configurable horizon (default 6 bars / 1 minute), in line with the IC distribution of the seed alphas.

### Running it

```bash
# default: 2 papers, 3 ideas, |IC@6| ≥ 0.01
python run_factor_research.py

# weekly-style session inside a 1-year sim
python run_factor_research.py \
    --session-id 2019-W03 \
    --cutoff 2019-01-21 \
    --n-papers 2 --n-ideas 3 \
    --ic-threshold 0.01 --horizon 6

# wipe all prior researcher factors first (start fresh)
python run_factor_research.py --reset
```

### Programmatic use (for the 1-year simulation orchestrator)

```python
from quant_fund_agent.agents.factor_research.graph import (
    factor_research_graph, reset_researcher_state,
)
from quant_fund_agent.agents.factor_research.state import FactorResearcherState

# At the start of each simulation: clean slate.
reset_researcher_state()

# Each weekly research session:
state = FactorResearcherState(
    session_id=f"2019-W{week:02d}",
    cutoff_date=week_start_date,
    n_papers=2,
    n_factor_ideas=3,
    ic_threshold=0.01,
    ic_target_horizon=6,
)
result = factor_research_graph.invoke(state)
new_factor_ids = result["kept_factor_ids"]
```

The `Selector` agent already reads `data/factors/factor_db.json` and will automatically see the new researcher factors on its next call — no plumbing changes needed downstream.

### Data context the agent is told about

`backtesting/data_loader.py` materialises **every** LOBSTER column into the panel.  All of these are available in the `data` dict passed to `BaseFactor.calc` and listed in the agent's prompt:

| Field        | Meaning                                                                          |
| ------------ | -------------------------------------------------------------------------------- |
| `open`       | start-of-bar mid-price (= `mid`)                                                 |
| `high`       | `max(mid, midEnd)` within the bar                                                |
| `low`        | `min(mid, midEnd)` within the bar                                                |
| `close`      | end-of-bar mid-price (= `midEnd`)                                                |
| `volume`     | `abs(trade)` — legacy alias for unsigned traded volume                           |
| `trade`      | signed traded volume in the bar (+buyer-initiated, -seller-initiated)            |
| `orderFlow`  | signed top-of-book limit-order flow (shares)                                     |
| `hidden`     | hidden traded volume                                                             |
| `auction`    | auction-print volume                                                             |
| `spread`     | end-of-bar quoted bid-ask spread                                                 |
| `effSpread`  | volume-weighted effective spread (sparse — only on bars that printed)            |
| `lobImb`     | top-of-book LOB imbalance `(bid-ask)/(bid+ask)` in [-1, +1]                      |
| `effLobImb`  | depth-weighted effective LOB imbalance (sparse — only on bars that printed)      |
| `trdLiq`     | trade-side liquidity proxy (size per unit price move)                            |
| `ofLiq`      | order-flow liquidity proxy (size posted per unit price move)                     |
| `depth`      | average top-of-book depth (best bid + best ask sizes)                            |
| `nbEvents`   | number of LOB events in the bar                                                  |
| `nbHidden`   | number of hidden trade prints                                                    |
| `nbTrades`   | number of visible trade prints                                                   |

All fields share the same `DatetimeIndex` (10-second bars) and ticker columns, so cross-field arithmetic is safe.  Two fields are intrinsically sparse (`effSpread`, `effLobImb`), since they only exist on bars where trades actually printed; the prompt instructs the LLM to guard against this with `.fillna`, `.replace(0, np.nan)`, and `min_periods` on rolling ops.

### Memory model for the research-time backtest

A research session on a laptop is bound by RAM, not CPU.  `load_panel` is therefore designed around three layers of memory control, all on by default:

1. **`fields=[...]`** — only materialise the panel fields the candidate factors actually need.  The required set is computed automatically from each factor's declared `inputs = [...]` class attribute.
2. **`usecols=`** — `read_csv` only loads the *raw* CSV columns needed to build those fields, not all 19.  For a typical 6-field session this is a ~3× I/O / memory win per ticker.
3. **`n_tickers`** — caps how wide the panel is.  The agent defaults to **15 tickers** for the research-time IC backtest (the binding fix for `zsh: killed` OOMs on a MacBook Air), and downcasts everything to **`float32`** for another ~50 % cut.  The full universe is still available for the final 1-year backtest, where you'd run with `--n-tickers 0`.

Empirically, the new defaults bring the research-time panel build down from "kills the kernel on a 16 GB MacBook Air" to ~200 MiB peak RSS for 15 tickers × 6 fields × ~46 k bars.  The `pd.DataFrame(series_dict)` index-alignment step is the dominant cost, and it scales superlinearly in the number of tickers (different per-ticker DatetimeIndices have to be unioned and reindexed), which is why capping the universe — not just the fields — is what unblocks the run.

```python
# Library-level usage
panel = load_panel(
    "ticker_data",
    fields=["close", "orderFlow", "nbTrades"],
    n_tickers=15,
    dtype="float32",
)
```

```bash
# CLI: research with 15-ticker sample (default) vs full universe
python run_factor_research.py                       # 15 tickers
python run_factor_research.py --n-tickers 0         # ALL tickers (memory-heavy)
python run_factor_research.py --n-tickers 30        # explicit cap
```

## Portfolio Manager Agent  *(new)*

### What it does

The Portfolio Manager (PM) is the final layer of the stack.  By the time
it runs, the research agents have already produced an open-ended library
of approved strategies in the Strategy DB; the PM's job is to decide:

1. **Which strategies to deploy** from the universe.
2. **How much capital to allocate** to each one.
3. **Which deployed strategies to keep / pause / kill** based on live
   performance — flagged ones go back to the research team, retired ones
   are gone for good.

The PM is *not* a research agent.  It does not invent factors, run
backtests, or write Python — it consumes everything that came before it.

### Pipeline

```
                ┌────────────────────────────────┐
                │   PortfolioManagerState        │
                │ (mode, personality, profile,   │
                │  strategy_db, portfolio_db)    │
                └──────────────┬─────────────────┘
                               │
            ┌──────────────────▼──────────────────┐
            │   1. load_universe                   │
            │   • read strategy_db                 │
            │   • refresh correlation matrix       │
            │   • build covariance matrix          │
            └──────────────────┬──────────────────┘
                               ▼
            ┌─────────────────────────────────────┐
            │   2. monitor_performance             │
            │   for each LIVE strategy: keep /     │
            │   flag / retire (rule-based in       │
            │   SELECTOR mode, LLM in ACTIVE mode) │
            └──────────────────┬──────────────────┘
                               ▼
            ┌─────────────────────────────────────┐
            │   3. screen_strategies               │
            │   apply personality filters          │
            │   (min Sharpe, max DD, max ρ),       │
            │   greedy diversified pick OR LLM     │
            └──────────────────┬──────────────────┘
                               ▼
            ┌─────────────────────────────────────┐
            │   4. construct_portfolio             │
            │   SELECTOR: profile.default_method   │
            │   ACTIVE:   LLM picks the method     │
            │   (or weights directly via           │
            │    construction_method=custom_llm)   │
            └──────────────────┬──────────────────┘
                               ▼
            ┌─────────────────────────────────────┐
            │   5. finalise                        │
            │   • build PortfolioAllocation list   │
            │   • compute expected metrics         │
            │   • write PortfolioRecord to DB      │
            │   • update each StrategyRecord:      │
            │       LIVE / PAUSED / FLAGGED /      │
            │       RETIRED                        │
            └─────────────────────────────────────┘
```

### Modes

| Mode       | Strategy selection                              | Construction method                                                    |
| ---------- | ----------------------------------------------- | ---------------------------------------------------------------------- |
| `SELECTOR` | Rule-based: top-N by IS Sharpe, capped at the personality's max pairwise correlation, filtered by personality hard constraints. | Fixed: uses `profile.default_construction_method` (or `--method override`). |
| `ACTIVE`   | LLM-driven, with the personality as system context.  Falls back to the SELECTOR rule on LLM failure. | LLM-driven: it picks one of the seven construction methods, or even supplies weights directly with `custom_llm`. |

Pick `SELECTOR` for reproducibility (no LLM in the hot path; the same
strategy DB always gives the same portfolio).  Pick `ACTIVE` when you
want the PM itself to reason about which method best fits the current
regime.

### Personalities

Personalities are plain dataclasses in `portfolio/personalities.py` that
bundle screening filters + portfolio shape + a default construction
method.  Three are shipped:

| Personality   | min Sharpe | max DD | max ρ | target N | max wᵢ | default construction |
| ------------- | ---------- | ------ | ----- | -------- | ------ | -------------------- |
| `defensive`   | 0.8        | −5 %   | 0.70  | 12       | 0.15   | `min_variance`       |
| `balanced`    | 0.6        | −10 %  | 0.80  | 10       | 0.25   | `risk_parity`        |
| `aggressive`  | 0.4        | −20 %  | 0.90  | 6        | 0.40   | `max_sharpe`         |

Personalities also carry `live_sharpe_floor`, `live_sharpe_kill_floor`
and `live_dd_floor` — the thresholds at which the rule-based monitor
flags or retires a deployed strategy.

Multiple PMs are supported by running the agent multiple times with
different `pm_name` + `personality`; each writes its own
`PortfolioRecord`s into the shared `PortfolioDatabase`, distinguished by
`pm_name`.  There is no central state to coordinate.

### Construction library

`quant_fund_agent.portfolio.construction` is a pure-Python library — no
LangGraph, no LLM — that you can use independently of the PM agent:

```python
from quant_fund_agent.portfolio import construct_portfolio
from quant_fund_agent.schemas import ConstructionMethod

weights = construct_portfolio(
    method=ConstructionMethod.RISK_PARITY,
    strategy_ids=["s1", "s2", "s3"],
    cov=db.covariance_matrix(annualisation_factor=BARS_PER_YEAR),
)
```

| Method                       | Inputs needed              | When to use |
| ---------------------------- | -------------------------- | ----------- |
| `equal_weight`               | strategy IDs only          | Robust 1/N benchmark, always works.                                |
| `inverse_volatility`         | cov (diagonal only)        | Quick risk-balancing without trusting off-diagonals.                |
| `min_variance`               | cov                        | When return forecasts are unreliable.                               |
| `mean_variance`              | μ, cov, λ                  | Classical Markowitz; tune λ for the risk appetite.                  |
| `max_sharpe`                 | μ, cov                     | Tangency portfolio; the most common "active" choice.                |
| `risk_parity`                | cov                        | Multi-strategy default — every strategy contributes equal risk.     |
| `hierarchical_risk_parity`   | cov                        | Robust to ill-conditioned cov; useful with many strategies.         |
| `custom_llm`                 | weights from the LLM       | ACTIVE PM only; the LLM hands you a weight vector directly.         |

All long-only solvers carry a feasibility clamp: if
`max_weight_per_strategy × N < 1`, the cap is automatically relaxed to
`1/N` so the QP isn't infeasible.

**Why `max_sharpe` uses the Schaefer (1980) transformation.**  The
Sharpe ratio `(μ−r_f)ᵀw / √(wᵀΣw)` is scale-invariant in `w`, so its
gradient is orthogonal to the radial direction `w → α w`.  SLSQP's
line search then routinely fails with *"positive directional derivative
for linesearch"* — even at valid optima — and would silently fall back
to equal weight.  Instead we solve the equivalent convex QP

```
min  yᵀ Σ y     s.t.  aᵀy = 1,  y ≥ 0,  (optional) y_i ≤ ub · Σy ∀i
```

where `a = μ − r_f·1` and then recover `w = y / Σy` (Schaefer 1980).
This is a plain QP, has a unique minimiser when `Σ` is PD, and SLSQP
solves it reliably across the entire test grid (long-only, bounded,
highly-correlated pairs, near-equal Sharpes).  For the shortable case
(`allow_short=True`) we use the closed-form tangency `w = Σ⁻¹a /
(1ᵀΣ⁻¹a)` — no solver needed.

### Correlation / covariance — lazy, cached, event-invalidated

`StrategyDatabase` maintains three layers of cache so the PM agent can
be invoked frequently in a long backtest without paying for I/O or
correlation maths each time:

1. **`_returns_cache`** — in-memory copy of every per-strategy PnL
   series.  Populated lazily; each CSV is read from disk at most once
   per process.
2. **`_correlation_matrix`** — cached pairwise correlations.
3. **`_covariance_matrix_bar`** — cached bar-scale covariance.  The
   accessor multiplies by `annualisation_factor` on read so callers
   can ask for any time scale without breaking the cache.

A single boolean `_correlations_dirty` invalidates (2) and (3) together.
Operations are split by whether they affect the matrices:

| Operation                                                   | Sets dirty? |
| ----------------------------------------------------------- | :---------: |
| `add_strategy` / `register_strategy` / `update_strategy`    | ✅ |
| `save_returns` / `append_returns` / `remove_strategy`       | ✅ |
| `load_from_json`                                            | ✅ |
| `set_pm_status` / `flag_strategy` / `retire_strategy`       | ❌ |

The first `correlation_matrix` / `covariance_matrix` read after a
mutation triggers exactly one recompute; subsequent reads are O(1).
`refresh_correlations()` remains as an explicit force-refresh for
"the files changed on disk behind the DB's back" scenarios.

**Why this matters.**  In a year-long backtest with daily PM rebalances
the strategy book changes ~weekly but the PM is invoked ~daily.  In
that regime the old eager-recompute design did O(N² · T) work on every
PM call; the cache cuts this to O(1) on the steady-state days.  An
empirical benchmark on a 10-strategy DB with 600 bars shows **~3700×**
speedup on 50 repeat reads vs 50 forced refreshes (sub-millisecond vs
~1.2 s).  At per-bar rebalance frequencies, the difference is the
between minutes and many hours of total simulation time.

**Incremental appends.**  `StrategyDatabase.append_returns(strategy_id,
new_bars, persist_every_n=None)` is the canonical way to extend a
live strategy's PnL inside a simulation loop.  It updates the
in-memory series in O(K) (K = new bars) and only rewrites the on-disk
CSV every `persist_every_n` calls — useful when bars arrive one at a
time and disk I/O would otherwise dominate.  `flush_returns()` forces
a final write at the end of the simulation.

**When persisted returns are missing** (e.g. when an old
`strategy_db.json` is loaded with no `returns/` directory), the
covariance matrix falls back to a `vol × ρ × vol` reconstruction using
the summary volatility on each `StrategyRecord.backtest_metrics`.

**What we deliberately didn't do.**  We considered Welford-style online
updates that would shave the recompute itself from O(N² · T) to
O(N²) per bar.  We deferred it: the dirty-flag pattern already
eliminates ≥ 99 % of wasted work for weekly/daily rebalances, and
online updates introduce real complexity around per-pair valid-overlap
windows.  If profiling later shows the recompute itself is the
bottleneck (per-bar PM rebalancing on very long horizons), the
extension is non-breaking — the existing API stays the same.

### Strategy lifecycle

`StrategyRecord` carries two parallel status fields:

| Field        | Tracks                                                              |
| ------------ | ------------------------------------------------------------------- |
| `status`     | Research state: `draft → backtesting → approved → live → retired`.  |
| `pm_status`  | PM state: `not_deployed → live → paused / flagged_for_review / retired`. |

Once the statistician approves a strategy it enters the PM's universe as
`NOT_DEPLOYED`.  The PM may then promote it to `LIVE`, downgrade an
existing `LIVE` strategy to `PAUSED` when it is no longer selected for
the new allocation, `FLAG` strategies whose live performance drifts
from in-sample (sent back to the research team), or `RETIRE` strategies
whose live performance has irrecoverably broken.

### PM committee — many PMs, one consensus

The default workflow is "many PMs, many sleeves" — each PM writes its
own `PortfolioRecord` and operates an independent slice of the book.
That works when you want explicit risk-budgeting *across* personalities
(e.g. 50 % defensive sleeve, 50 % aggressive sleeve).

When you instead want several PMs to **converge on a single
allocation**, use `run_pm_committee`.  The committee:

1. Runs every PM in **`propose_only` mode** — each produces a
   `PortfolioRecord` proposal but does **not** mutate `strategy_db` or
   `portfolio_db`.
2. Aggregates the proposals with one of three voting methods:
   * `SIMPLE_AVERAGE`   – mean weight across PMs.  Deterministic.
   * `WEIGHTED_AVERAGE` – per-PM weights via `CommitteeConfig.pm_weights`
     (e.g. give a "chair" PM 5× the vote).
   * `LLM_MODERATOR`    – one LLM call reads every proposal + rationale
     and synthesises the consensus.
3. Resolves monitor consensus: a strategy is flagged / retired when at
   least `flag_threshold` / `retire_threshold` of the PMs vote that way
   (default 0.5 = simple majority).
4. Writes **one** `PortfolioRecord`, applies **one** wave of
   `pm_status` updates, and stores every per-PM proposal in
   `record.metadata["pm_proposals"]` for audit.

**Strategies the committee flags / retires are automatically excluded
from the consensus weights**, so the allocation list always reflects
what will actually trade.

**Single-PM passthrough.**  Calling `run_pm_committee([one_pm], ...)`
short-circuits the aggregation and is byte-for-byte identical to
invoking the PM directly — so existing single-PM behaviour is
unchanged.

```python
from quant_fund_agent.agents.portfolio_manager.committee import (
    CommitteeConfig, run_pm_committee,
)
from quant_fund_agent.schemas import VotingMethod

# Three PMs, simple-average consensus, single PortfolioRecord written.
record = run_pm_committee(
    [defensive_state, balanced_state, aggressive_state],
    CommitteeConfig(
        voting_method=VotingMethod.SIMPLE_AVERAGE,
        pm_name="board",
    ),
    strategy_db,
    portfolio_db,
)

# Same PMs but the LLM chairs the discussion.
record = run_pm_committee(
    [defensive_state, balanced_state, aggressive_state],
    CommitteeConfig(
        voting_method=VotingMethod.LLM_MODERATOR,
        pm_name="board_llm",
    ),
    strategy_db,
    portfolio_db,
)
```

### Running it

```bash
# SELECTOR mode, balanced personality, default construction (risk parity).
python run_portfolio_manager.py

# Pick a different personality and override the construction method.
python run_portfolio_manager.py --personality defensive --method min_variance --target-n 12

# Same strategy DB, two PMs running side-by-side with different sleeves.
python run_portfolio_manager.py --pm-name defensive_sleeve  --personality defensive
python run_portfolio_manager.py --pm-name aggressive_sleeve --personality aggressive

# Three PMs, ONE consensus portfolio (simple-average voting).
python run_portfolio_manager.py --committee defensive,balanced,aggressive

# Same committee but the LLM chairs and synthesises the consensus.
python run_portfolio_manager.py --committee defensive,balanced,aggressive \
    --committee-voting llm_moderator

# ACTIVE mode — the LLM decides the construction method itself.
python run_portfolio_manager.py --mode active --personality balanced
```

Each run reads `data/strategies/strategy_db.json`, refreshes
correlations, invokes the PM agent, prints the allocation, and appends a
`PortfolioRecord` to `data/portfolio/portfolio_db.json`.

### Programmatic use

```python
from quant_fund_agent.agents.portfolio_manager.graph import portfolio_manager_graph
from quant_fund_agent.agents.portfolio_manager.state import PortfolioManagerState
from quant_fund_agent.databases import PortfolioDatabase, StrategyDatabase
from quant_fund_agent.schemas import PMMode, PMPersonality

strategy_db = StrategyDatabase()
strategy_db.load_from_json("data/strategies/strategy_db.json")
portfolio_db = PortfolioDatabase()

result = portfolio_manager_graph.invoke(PortfolioManagerState(
    pm_name="balanced_pm",
    mode=PMMode.SELECTOR,
    personality=PMPersonality.BALANCED,
    target_n_strategies=10,
    strategy_db=strategy_db,
    portfolio_db=portfolio_db,
))
record = result["portfolio_record"]
print(record.construction_method, record.allocations, record.expected_metrics)
```

### Why this design

- **Single source of truth.**  Every PM decision becomes a
  `PortfolioRecord`.  Multiple PMs in parallel just produce parallel
  streams, distinguished by `pm_name`.
- **LLM is optional.**  SELECTOR mode is fully deterministic; ACTIVE
  mode only adds the LLM where reasoning is genuinely useful (method
  choice and live monitoring).  The LLM path also falls back to the
  rule-based path on any failure so the agent never crashes mid-run.
- **Pure-Python optimisers, no `cvxpy`.**  All seven construction
  methods are built on `scipy.optimize` (already a dependency); for a
  multi-strategy fund with ≤ a few dozen strategies the QPs solve in
  microseconds.
- **Pairwise correlation lives on the records.**  The PM agent doesn't
  re-compute correlations on every call — it reads them off
  `StrategyRecord.correlations`, kept in sync by
  `StrategyDatabase.register_strategy` and `refresh_correlations`.
- **Strategy lifecycle has two axes.**  The research lifecycle
  (`status`) and the PM lifecycle (`pm_status`) are kept independent.
  The statistician's approval doesn't mean "deploy"; the PM's "retire"
  doesn't mean "throw away the research record".

## MCP tooling — every agent's toolbox is an MCP server

Following Anthropic's Model Context Protocol convention, **every agent calls its
deterministic toolbox over MCP**.  The agents themselves stay synchronous
LangGraph nodes whose *reasoning* is still a plain `llm.invoke()`; what moved
behind MCP is the heavy / deterministic work each agent does (data loading,
backtests, code materialisation, statistical tests, portfolio maths).  Each MCP
server owns its heavy data server-side (e.g. the factor panel is loaded once and
cached) and only small JSON crosses the boundary.

| Server (`quant_fund_agent/mcp/…`) | Used by        | Tools                                                            |
| --------------------------------- | -------------- | ---------------------------------------------------------------- |
| `modeling_server`                 | Architect      | `list_models`, `fit_and_backtest`                                |
| `catalog_server`                  | Selector       | `load_factor_catalog`                                            |
| `research_server`                 | Factor Researcher | `load_papers`, `existing_factor_ids`, `materialise_factor`, `backtest_factors`, `persist_results` |
| `statistics_server`               | Statistician   | `list_tests`, `run_tests`                                        |
| `portfolio_server`                | Portfolio Mgr  | `screen_strategies`, `construct_portfolio`, `expected_portfolio_metrics` |

Each server has a sibling `*_client.py` (the synchronous facade the agent calls)
and a `*_service.py` (the shared implementation).  A single
`quant_fund_agent/mcp/_bridge.py` runs one persistent stdio session per server on
a shared background event loop.

**In-process fallback.**  Every client falls back to calling the same
`*_service` functions in-process — identical results, no subprocess.  This is
used by the test suite and as an escape hatch.  Toggle MCP off globally with
`QF_USE_MCP=0`, or per server:

```bash
MODELING_USE_MCP=0  CATALOG_USE_MCP=0  RESEARCH_USE_MCP=0  \
STATISTICS_USE_MCP=0  PORTFOLIO_USE_MCP=0
```

Parity tests in `tests/test_mcp_*.py` assert the MCP path and the in-process path
return identical results for each server.

Database reads/writes (the in-memory `StrategyDatabase` / `PortfolioDatabase`
handles that flow through graph state and the PM committee) stay in the agent
graphs — they hold live Python objects that cannot cross a stateless MCP
boundary; only the pure computations are exposed as tools.

## Getting Started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your API keys

# 1. (One-off) backtest every seed factor and persist factor_db.json
python run_all_factors.py

# 2. Whole fund in one command: (optional research) → strategy pipeline →
#    PERSIST approved strategies to strategy_db.json → Portfolio Manager.
#    This is the end-to-end path and the basis for the upcoming backtest.
python run_fund.py --n-strategies 3

# ── or drive the stages individually ──
python run_factor_research.py --n-papers 2 --n-ideas 3   # Factor Researcher
python run_pipeline.py                                   # Selector→Architect→Statistician (no persist)
python run_portfolio_manager.py                          # PM over the persisted book
python -m quant_fund_agent.main                          # single-task orchestrator router
```

### How a strategy reaches the Portfolio Manager

The statistician *judges* a strategy; it does not persist it.
`quant_fund_agent.pipeline.persist_approved_strategy` turns an accepted
candidate into a `StrategyRecord` (in-sample + out-of-sample metrics) and
registers it **with its per-bar PnL series** via
`StrategyDatabase.register_strategy`.  That returns series is what the
cross-strategy correlation matrix and PM allocation are built on — without
this step the Strategy DB stays empty and the PM has nothing to allocate.
`run_fund.py` and `demo_pipeline.ipynb` both go through this path; the same
functions are designed to be called on a weekly schedule inside the
two-month backtest.
