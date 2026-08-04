# QuantFundAgent

Most of this detailed README is AI-generated. A short overview of the different folders and interesting files:

The core logic and all agents sit within the quant_fund_agent folder. The databases (the factors, papers and strategies) sit within the data folder. The diagrams folder contains diagrams of the system architecture as well as the architecutre of every single agents as both .png and .excalidraw files. prelim_files and tests contain tests or other files that have been created earlier and are not relevant to the current state of the project, but might still be of future interest for the thesis. 

The notebook demo_pipeline.ipynb demonstrates the different agents and there outputs. I preran all cells with my API keys and the data I downloaded. Rerunning them will not work, since it requires API keys as well as the data, which I can not push to Github because the files are too large. 

**notebooks/evolution_walkthrough_for_supervisor.ipynb** is the guided walkthrough of the evolutionary factor researcher (knowledge-graph snapshot → graph-steered seeding → codegen → the four-axis Pareto scoring → reflection → mutation → one live generation → real surviving factors from the thesis runs). Unlike demo_pipeline.ipynb it IS rerunnable from a fresh clone: the knowledge graph, paper index, embeddings and the L4_terra_s0 run artifacts are committed, market data comes from the free yfinance path, and the only thing needed is an OpenAI key in `.env` (a full pass costs ~$1–2; outputs of a completed pass are saved in the notebook). 

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
- **Evolutionary factor researcher** — `run_factor_evolution.py` performs deterministic constrained NSGA-II search over four maximised objectives: `marginal_value`, `independence`, `parsimony`, and `structural_novelty`. Diversity is organised by a two-level hierarchy: knowledge-graph mechanism groups reserve coverage of distinct economic communities, while multiple classic demes evolve independently inside each group and exchange elites only within that group. An explicit low-probability cross-group crossover can synthesize coherent interactions. The former Quality-Diversity grid and `temporal_robustness` Pareto axis are removed; temporal degradation remains a reflection diagnostic, not a selection objective or hard gate. The accepted book is the union of the per-group Pareto archives — optionally bounded by a per-group **archive cap** (`--archive-cap`, crowding-distance cull with eviction lineage events) — with optional end-of-run curation and publish-time multiple-testing deflation (both now fail **closed**; `QF_PERSIST_FAIL_OPEN=1` restores the legacy fallback with a metadata flag). The LLM layer (`quant_fund_agent/llm.py`) is multi-provider (OpenAI / Anthropic / **Amazon Bedrock** via `anthropic.`-prefixed ids / any OpenAI-compatible endpoint via `FACTOR_RESEARCH_LLM_BASE_URL`) and carries a per-role **token/cost meter** with a hard `--max-cost-usd` ceiling, optional full prompt/response transcripts (`QF_LLM_TRANSCRIPT_PATH`), and provenance stamped into each prerun's manifest. Progressive reveal is instrumented for drift analysis (`reveal_index`-stamped `prequential.jsonl`, per-member `rescore` diagnostics + `rescore_failed` rows, per-generation `gen_quality.jsonl`) and accelerated by a byte-exact combined-fit cache. A small `--creative-frac` mixes explicitly-ungrounded, novelty-encouraged hypotheses into grounded runs, and `--memory-key` isolates experience memory per ablation arm. **The full thesis comparison run (ablation ladder + 6-model sweep) is specified in `docs/research-evolution/FINAL_RUN_PLAN.md`** and orchestrated by `run_ablation_matrix.py` + `matrix/final_matrix.yaml` (per-entrypoint defaults so GP/oneshot arms never inherit evolution-only flags; guarded by `tests/test_final_matrix_plan.py`). `--mechanism-groups-mode max` makes the requested group count a hard upper limit — the run uses however many usable mechanism communities the knowledge graph actually forms. `scripts/rebuild_embeddings.py` re-embeds the paper corpus (runbook step 2); `--n-tickers 0` means the full universe on every entrypoint (evolution, GP, oneshot, timing probe). A **non-evolutionary `--variant refine` ablation** (2026-07-31) swaps only the operators: each seeded factor is refined by the LLM against its own deterministic evaluation report at most `--refine-rounds` times (same factor, same mechanism — `build_refine_prompt`), demes continuously re-seed fresh graphrag-grounded ideas, and the only combination operator is the occasional cross-group synthesis; scoring, gates, Pareto axes, progressive reveal, curation and deflation are identical (resume-safe `refine_state.json`; `matrix/terra_l4_refine.yaml` runs it as `L4R_terra_s0` against the finished evolutionary `L4_terra_s0`). **The walk-forward Terra ladder (launched 2026-08-01, `matrix/terra_wf_ladder.yaml`)** re-runs the ablation ladder on GPT-5.6 Terra with a two-phase progressive-reveal schedule (`--wf-blocks 10 --wf-block-bars 126`): the panel extends to 2026-07 with no forward reserve or test tail — instead the final 10 generations each reveal one ~6-month block of 2021→2026 that is prequentially scored (traded) *before* the archive may adapt to it, yielding a 5-year live walk-forward record per arm; `--graph-readonly` freezes the knowledge-graph snapshot so every arm resolves identical mechanism groups, and the L1 oneshot baseline researches only on the pre-walk-forward span (`quant.config.nasdaq100_2010_to2021.yaml`). **Rescore-cost hardening (2026-08-03):** the every-generation archive rescore (which had grown to 9h+/generation as archives and the revealed window expanded) now shares one canonically-ordered whole-archive "with" fit across all members, assembles fit matrices from cached per-signal standardised feature columns, and skips the jitter/reference probes during rescore — carrying each member's admission-time plateau dock forward (`plateau_penalty_carried`) instead of re-running its jittered code variants; admission-time child evaluation is unchanged. **Factor-book cross-analysis (2026-08-04):** `scripts/analyze_l4_factor_books.py` compares the finished `L4_terra_s0` and `L2_opus5_s0` books factor-by-factor against the 101 formulaic alphas (per-factor + combined lightgbm IC on DEV/TEST/FORWARD, correlation structure, effective factor counts, zoo-novelty) with figures + REPORT.md under `data/comparisons/l4_factor_analysis/` — headline: the Opus book generalises best OOS (combined TEST IC 0.045), the Terra book adds the most marginal/diversity value on top of the formulaic zoo, and the two books are near-orthogonal (mean |rho| 0.056).
- **Non-LLM GP factor-mining benchmark** — a deterministic **genetic-programming** alpha miner (`run_gp_factor_mining.py`, `agents/factor_research/gp/`) as a no-LLM baseline for the evolutionary researcher (AutoAlpha spirit: hierarchical evolutionary mining of formulaic alphas; AlphaGen's "score a combined set" folded in). It mines **typed expression trees** over the project's **base grammar** (`factors.ops.BASE_OPS` + arithmetic) via subtree crossover + subtree/point/hoist mutation and **hierarchical depth-schedule growth** (`--depth-schedule 3,5,7`), and **reuses the LLM arm's NSGA-II controller, `evaluate_fitness` scoring seam and `persist_archive` verbatim** — so the two are apples-to-apples (only child *proposal* differs: GP vs LLM). Crucially the GP is **confined to the base grammar** (operators drawn only from the explicit `BASE_OPS` tag, never `dir(ops)`), whereas an LLM factor may *extend* the grammar (inline helpers / scipy·sklearn·numpy·pandas) — a deliberate agentic advantage the benchmark is denied. Two further asymmetries follow from the GP having no knowledge graph and should be stated when reporting the ablation: it runs a **single** mechanism group of `--islands` flat demes rather than the LLM arm's two-level hierarchy, and `--progressive-reveal` is LLM-arm only (run the LLM arm without it to put both arms on identical splits). Runs in-process (no server, no API key), persists a normal prerun (`source=researcher`, `engine=gp`; main seed DB untouched), and drops straight into `run_model_comparison.py --preruns <gp>,<llm>` as the non-LLM baseline row. Design: `docs/research-evolution/GP_BENCHMARK.md`.
- **High-frequency single-name runs** — the same L4WF walk-forward arm runs on **10s LOBSTER bars for one ticker** (first run: GLD, `quant.config.gld_hf.yaml`, prerun `lobster_equity_gld_hf/L4WF_gld_s0`, launched 2026-08-03): per-underlying IC makes single-ticker fitness well-defined, retrieval auto-masks to price/general papers when no fundamental field is in scope, and graphrag's gap query keeps only mechanisms computable from the LOBSTER fields. `QF_EXECUTION_LAG_BARS` delays every forward-return label by L bars (`close[t+L+h]/close[t+L]−1`, run uses 3 bars = 30 s) so signals are never assumed to trade on the bar they're computed; `QF_SIGNAL_CACHE_MAX` bounds the eval signal cache on small-RAM hosts; `LobsterProvider` sniffs CSV headers so only book levels actually on disk are advertised to the researcher. Ops harness: `scripts/gld_overnight_supervisor.sh` (crash-relaunch with checkpoint resume, RSS watchdog, chained post-run `run_model_comparison.py` IC + strategy backtest).
- **Landing-page example generator** — `showcase_pipeline/landing_examples/` turns real pipeline runs into provenance-stamped marketing artifacts for the startup spinoff (exported to `../company-brain/marketing/examples/`). `run` drives Selector→Architect→Statistician attempts and dumps **every** candidate (approved *and* rejected — the "Likely overfit" showcase card is a rejected candidate, which `run_fund.py` never persisted) with full trial history + stat-test details; `list` tables each candidate's harness metrics including a **per-strategy CSCV probability of backtest overfitting** (`pbo_cscv` over the return series of the variants the Architect actually tried) and a **deterministic badge** (Robust / Worth testing / Likely overfit — fixed thresholds on deflated-Sharpe/PBO/OOS, never an LLM's judgement); `export --pick` writes per example a `card.json`, stitched IS/OOS equity curves (raw + SVG-ready polylines), a `card.png`, a "Behind the verdict" markdown teardown (idea → generated factor code → gates → deflation arithmetic), a grounded chat transcript and a `provenance.json` (config hash, git commit, OOS-recompute cross-check). All published copy is template-generated with a banned-word compliance guard. Tests: `tests/test_landing_examples.py`.

> **Current default search shape:** 5 knowledge-graph mechanism groups × 3 demes per group, population 16 per deme, 12 generations, 4 proposed children per deme per generation, 6 seed ideas per group, within-group migration every 3 generations, and 10% explicit cross-group synthesis. Use smaller values explicitly for smoke tests.

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
#    data/comparisons/<id>/.  Four tracks: single-factor IC, factor analytics
#    (diversity/redundancy + deflation/importance), ML-combined-signal vectorised
#    backtest, downstream agents.  Only downstream spends LLM; --no-downstream is
#    a fully offline run.
./venv/bin/python run_model_comparison.py --preruns gpt4omini,claude
QF_USE_MCP=0 ./venv/bin/python run_model_comparison.py --all --no-downstream

# Fast + crash-safe on the big intraday panel: --fast strides the panel to
# 20000 bars (override with --max-bars), and every track is checkpointed to
# disk as it finishes (status.json), so an interrupt never loses progress.
./venv/bin/python run_model_comparison.py --preruns gpt4omini,claude \
  --data-dir ticker_data --no-downstream --fast

# Pick the exact underlyings and split IS/OOS by the calendar (train on three
# months, test on the next) instead of by ticker-count / tail-fraction.  Each
# window is a comma list of months/dates or an inclusive START:END range; they
# must be disjoint, and the panel is restricted to their union.
./venv/bin/python run_model_comparison.py --all --no-downstream \
  --tickers AAPL,MSFT,CORN \
  --train-months 2024-06:2024-08 --oos-months 2024-09
```

The **factor-analytics** track (`comparison/analytics.py`) is near-free — it reuses
the factor signals already cached by the IC track — and answers questions IC and the
ML track can't: is a zoo genuinely *diverse* or just redundant variants
(`eff_n_factors`, `eff_ratio`, `n_clusters`), and is its best |IC| real or a
multiple-testing artefact (`deflated_best_ic`, `deflated_best_t`), plus which factors a
model actually leans on (LASSO/GBM importance, `lasso_sparsity`).

Single factors are scored by **IC**. By default (`--fit-standardize per_underlying`) this is a
**per-underlying time-series IC** — one Pearson correlation per asset between the
factor's value vector and that asset's *own* forward-return vector, aggregated as a
valid-observation-weighted mean — so it is meaningful for a single ticker and has
*no cross-section* (pass `--fit-standardize
cross_sectional` for the legacy across-tickers IC). Each factor is scored at *its own*
`prediction_horizon` (the `ic_own`/`horizon_own` columns + a `mean_abs_ic_own` summary)
alongside the shared 1/6/60-bar grid. The analytics diversity/importance
track uses the same per-underlying standardisation. The **ML track** likewise
treats a factor zoo as inputs to a model: each catalog model + an ensemble fits the factors
(standardised per-underlying over time on the in-sample window) into **one combined signal**
and that signal is run through a **simple per-underlying vectorised backtest**, fit on IS
and evaluated OOS — *not* a cross-sectional ranking, so it works with any number of
underlyings (even one). To keep the Sharpe unbiased, each bar's target position is held as
a **staggered "tranche" book** (the live position is the rolling mean of the last
`--holding-period` targets) and **marked to market on the 1-bar forward return**, rather
than multiplying a raw position by an `h`-bar forward return — which overlaps `h−1` bars
between adjacent rows and inflates the annualised return ~`h`× and the Sharpe ~`√h`×. This
is the same non-overlapping convention the deployed backtester uses. The combined signal's
forecast horizon is **derived from the constituent factors' own horizons** — the **mode** by
default (`--combined-horizon-agg {mode,median,max,min,explicit}`); `--horizon` is now an
*override* that forces it. Every other modelling choice is a CLI argument: `--holding-period`
(tranche length, default = the derived horizon),
`--position-mode {threshold,sign,continuous}` (default threshold band, `--position-threshold`),
`--position-zscore {expanding,full,rolling,none}`, `--aggregation {portfolio,per_underlying}`,
`--fit-standardize {per_underlying,cross_sectional}`.

Factors that need data not yet downloaded (e.g. fundamentals) are filtered and
reported, so the comparison runs on the current data and re-runs unchanged once
the full LOBSTER universe / FMP membership is in place.  `run_model_comparison.py
--research --prerun-spec name=model[:provider] …` can also mine the preruns first.

#### Rolling-window sweep over many tickers

`run_rolling_comparison.py` automates the comparison **per ticker over a rolling
IS/OOS month window** and aggregates everything. For each ticker under
`ticker_data/` it runs `run_model_comparison.py` on 2 IS months + the next OOS
month, stepping one month forward (the prior OOS month becomes the second IS
month), comparing the preruns **per underlying**. Each (ticker, window) runs in
its **own subprocess**, so memory is reclaimed between runs — no OOM on the large
intraday panel — and the sweep is **resumable** (completed windows are skipped)
and robust (a failed run is logged; the sweep continues).

```bash
# Whole sweep: all tickers, full resolution, all models (LLM-free).
./venv/bin/python run_rolling_comparison.py

# Quick smoke on one ticker, capped for speed.
./venv/bin/python run_rolling_comparison.py --tickers CORN --max-bars 5000 --name smoke

# Re-build combined tables + figures without re-running anything.
./venv/bin/python run_rolling_comparison.py --name smoke --aggregate-only
```

Output under `data/comparisons/<batch>/`: `combined/{bruteforce,importance,
diversity,ic}_all.csv` (every run, tagged with `ticker, oos_month, is_window`),
`per_ticker/<ticker>/importance_over_months__<prerun>__<model>.csv` + heatmaps
(how the **most important features change over the OOS months**) and
`performance_<metric>__<model>.png`, **cross-ticker figures** under
`cross_ticker/` (per factor set: OOS Sharpe / OOS IC / OOS÷IS-Sharpe ratio over
the OOS months as **one coloured line per ETF**, plus a **mean-OOS-Sharpe vs
average-daily-volume** scatter with one dot per ETF coloured by sector), plus
`summary.md` and a `manifest.json`.

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

**Codegen:** `agents/factor_research/codegen.py` validates generated code (AST checks, import allow-list, a required positive-int `prediction_horizon`, smoke test on synthetic data) before registering factors.

**Prediction horizon:** every factor carries its own forecast horizon (in *bars*) — `prediction_horizon` (+ optional `suggested_horizons`), chosen by the researcher (the prompts state the feed's bar size, inferred from the panel index, so it reasons in wall-clock time) and stamped on `FactorRecord`. Each factor's IC is anchored at *its own* horizon. Existing factors were backfilled to 6 (`scripts/backfill_horizons.py`).

**Seed vs researcher factors:** seed alphas live under `quant_fund_agent/factors/` (version-controlled). Researcher output goes to `factors/researcher/` (gitignored) and is tagged `source=RESEARCHER` in the factor DB. `purge_researcher_factors()` clears researcher state between simulation runs.

**Data scope:** the researcher only invents factors the configured feed can serve. `pipeline.run_research_session` computes `data.usable_fields(settings)` — the provider's advertised fields, honoring the LOBSTER **order-book level** (`lobster_level` 2 vs 3, set in `setup`) and the **fundamentals** opt-out — and threads it onto `FactorResearcherState.allowed_fields`. The brainstorm/codegen data-context (`prompts.build_data_context`) lists only those fields, and `filter_and_persist` drops any factor whose `inputs` fall outside the scope.

**Lookahead:** paper selection respects `published_date`; the IC backtest panel can be sliced to a `cutoff_date`.

### Selector

Loads the factor catalog, asks an LLM to formulate a trading hypothesis, then selects factors that best support it.

**Graph:** `agents/selector/graph.py`

### Architect

Turns the Selector's hypothesis into a tradable strategy in a **refinement loop**:

1. **Design** — pick model type, hyperparameters, position settings, and the **forecast horizon** from the modeling catalog. The Architect is shown each selected factor's own `prediction_horizon` and chooses `target_horizon` for the combined signal (default: the mode of those horizons); it persists across revise iterations.
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

# yfinance S&P100 (static list), monthly meetings, NO LLM research — instead inject
# 2 random factors/month from two preruns (seeds always available):
./venv/bin/python run_backtest.py --config quant.config.sp100.yaml \
  --start 2016-01-01 --end 2026-06-01 \
  --warmup 12M --grid-freq 1M --research-every 1M --strategy-every 1M --pm-every 1M \
  --factor-source prerun_inject \
  --inject-preruns sp100-5.4-mini,sp100-4o-mini --factors-per-meeting 2 \
  --fallback-spread-bps 2.0 --run-id sp100_inject
```

**Factor source — LLM research vs. prerun injection** (`--factor-source`):

| Mode | What fires on each `research-every` meeting |
|------|---------------------------------------------|
| `research` (default) | an LLM Factor-Researcher session invents factors as-of the cutoff |
| `prerun_inject` | **no LLM** — draws `--factors-per-meeting` (default 2) factors from the union of `--inject-preruns` (pre-filtered to those computable on the live panel), appended to the run catalog; seeds stay available, the pool draws without replacement |

`prerun_inject` reuses factors already mined into
`data/workspaces/<config>/preruns/<prerun>/`, so a long backtest grows its factor
universe over time exactly as periodic research would, but reproducibly and at zero
LLM cost. API-provider backtests are **config-driven**: `--config
quant.config.<x>.yaml` sets the provider / universe / timespan before the panel
loads (`quant.config.sp100.yaml` ships a yfinance S&P100 config from 2016).

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

**Outputs** land in `data/backtests/<run_id>/`: `equity.csv` (fund return, NAV,
drawdown, gross exposure, names held), `fund_metrics.json` (Sharpe / Sortino /
Calmar / maxDD / cost / turnover / % invested / final NAV), `attribution.csv`
(per-strategy contribution), `meetings.jsonl` (each meeting's injected factors /
approvals / allocation), the run's `config.json` + strategy/portfolio DBs (scoped to
the run, so the live book is never clobbered), and a **`report/`** folder with
presentation-ready figures (NAV + drawdown, cumulative return, percent invested &
names held, rolling Sharpe, monthly returns, per-strategy attribution, catalog/
strategy growth) and a `report.md` KPI table. `fund_showcase.ipynb` §11 renders the
latest run.

## Data

**You don't need the author's LOBSTER data to run this.** Data loads through a
pluggable provider layer (`quant_fund_agent/data/`); pick one with the guided
wizard:

```bash
cp .env.example .env                  # add OPENAI_API_KEY (+ FMP/AV keys if used)
python -m quant_fund_agent.setup      # choose provider/universe/timespan → quant.config.yaml
./venv/bin/python run_fund.py --n-strategies 1
```

The wizard also pins **what data the Factor Researcher may use**: for LOBSTER it asks the **order-book level** (`--lobster-level` 2 = book only + volume, 3 = full message stream incl. trades/order-flow/hidden), and for the API vendors whether to use **fundamentals** (`--fundamentals yes|no`, equity FMP/AlphaVantage). The researcher then only proposes factors the chosen scope can serve.

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

**Survivorship-bias-free universes (S&P 500, point-in-time).** A static ticker
list over a multi-year backtest over-represents survivors. Set `data.membership:
sp500` (or `QF_MEMBERSHIP=sp500`) and the universe becomes **time-varying**: the
provider fetches every name that was *ever* an S&P 500 member in your window (834
distinct tickers since 2010 vs ~503 today), and the panel is **masked per-bar** to
each date's actual constituents at load — so research, the architect/statistician,
the walk-forward trade loop and the comparison harness are all survivorship-correct
at once. The membership table is reconstructed from **free public sources**
(GitHub `fja05680/sp500` as primary + Wikipedia change-log as cross-check),
audited (month-end count 497–506; TSLA/ATVI spot-checks), and fully reproducible —
`./venv/bin/python scripts/build_sp500_membership.py`. Free path is **tickers-only**
(yfinance can't serve most delisted names — the residual gap the premium path
below closes); see [`SP500_MEMBERSHIP.md`](docs/data-layer/SP500_MEMBERSHIP.md).

**Premium path — a local FMP archive back to 2004.** With an FMP **Premium** key,
`scripts/fmp_bulk_download.py` pulls a one-time, resumable, local archive
(`data/vendor/fmp/`, gitignored) covering **every** name that was ever an S&P 500
or Nasdaq-100 constituent since 2004: adjusted *and* unadjusted OHLCV, dividends,
splits, daily market cap, and the full fundamental record (~300 raw fields per
quarter across income statement / balance sheet / cash flow / ratios /
key-metrics / growth / enterprise values / earnings). Two things follow:

- **delisted names actually load** — the residual survivorship bias of the free
  path is closed, and `symbol_map.csv` reports per-ticker *spell coverage* so
  what remains missing is measured rather than assumed;
- **~130 canonical fundamental fields** (up from 13) are exposed to the Factor
  Researcher through `provider: fmp_archive`, which reads the archive **offline**
  and stamps every value point-in-time — including the fix that `ratios` /
  `key-metrics` / `financial-growth`, which carry no filing date, inherit the
  matching income statement's *actual* `filingDate` instead of a flat 60-day lag.

Membership itself becomes FMP-native (`scripts/build_fmp_membership.py`, backward
walk of `historical-*-constituent`), audited and reconciled month-by-month against
the preserved free reconstruction. Run it with
`--config quant.config.fmp_sp500.yaml`; see
[`FMP_PREMIUM_ARCHIVE.md`](docs/data-layer/FMP_PREMIUM_ARCHIVE.md).

**LOBSTER specifics:** place CSVs under `ticker_data/` (or set `DATA_DIR`). The
loader (`backtesting/data_loader.py`) builds an aligned panel of OHLCV plus
microstructure fields (`orderFlow`, `lobImb`, `spread`, `nbTrades`, etc.) on a
shared 10-second index, **including per-level book columns** when present
(`askPrice{i}`/`askDepth{i}`/`bidPrice{i}`/`bidDepth{i}` — the price and displayed
depth at each order-book level, discovered from the CSV header so any level count
works).

**Building `ticker_data` from LOBSTER.** `quant_fund_agent/data/lobster_ingest/`
converts LOBSTER exports into the 10s-bar format above — levels-agnostic, one day
at a time (RAM-bounded), clipped to the regular session (2340 bars/day). It
auto-detects the product: **raw** message+orderbook (the 19 aggregate columns) or
the **sampled** 10s order book (book columns only; flow columns empty); both also
emit the per-level price/depth columns (`4·NumLevels` of them). Convert a folder
or a `.7z` you already have, offline:

```bash
./venv/bin/python scripts/convert_lobster.py --raw-dir GLD_2026-06-01_2026-06-11_3
```

For a multi-ticker, multi-year pull, `scripts/run_lobster_ingest.py` drives the
LOBSTER web portal end-to-end (validated live): a one-time `--recon` login, then
`--place-raw` requests the **raw** message+orderbook product (the single
`requestdata.php` form → the 19 aggregate columns + per-level book columns; the
bulk form returns only a book), and
`--ingest` downloads each finished `.7z` off `mydata.php` and **stream-converts it
day-by-day then deletes it** — so a 2-year raw archive never unzips in full.
Resumable via `orders_done.json`. The derived microstructure columns are
**reconstructed** from the LOBSTER spec (see the drift-warning in
[`docs/lobster-ingestion/`](docs/lobster-ingestion/README.md)).

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
