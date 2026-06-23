# Important Runnable Files

> **Modularisation by (config, prerun).** Researched factors and the strategies
> built from them are isolated per **(data config, research-LLM prerun)** under
> `data/workspaces/<config>/preruns/<prerun>/` (factors, strategies, returns,
> model artifacts, portfolio, showcase).  The canonical *main* factor library
> `data/factors/factor_db.json` holds **only the seed/formulaic alphas** and is
> never written by research.  `run_merge.py` composes a separate *active book*
> under `data/books/<name>/` by pooling chosen scopes for the PM; main is never
> mutated.  Run `scripts/migrate_main_seed_only.py` once to split the legacy
> mixed main DB into seed-only main + a preserved `legacy` scope.

---

## Setup

### `python -m quant_fund_agent.setup`
Guided wizard that writes `quant.config.yaml` — required before running anything that fetches market data.

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | `yfinance` | `yfinance` / `fmp` / `alphavantage` / `lobster` |
| `--asset-class` | provider default | `equity` / `crypto` / `fx` |
| `--freq` | `1d` | Bar frequency (`1d`, `1h`, `5m`, `1m`) |
| `--start` / `--end` | 2 years / today | Date range |
| `--preset` | `demo` | Universe preset (e.g. `demo`, `crypto_demo`, `fx_demo`) |
| `--tickers` | — | Comma-separated tickers (overrides `--preset`) |
| `--n-tickers` | — | Cap universe size |
| `--data-dir` | `ticker_data` | LOBSTER provider only |
| `--lobster-level` | `3` | LOBSTER only: order-book level the feed carries — `2` = order book + traded volume, `3` = adds the message-stream fields (trades / order-flow / hidden / auctions). |
| `--fundamentals` | `yes` | Equity FMP / AlphaVantage only: use non-OHLCV fundamental / estimate / event data. |
| `--cache-dir` | `data/market` | Parquet cache location |
| `--assist "<text>"` | — | LLM drafts a config from plain English |
| `--no-validate` | off | Skip validation fetch |
| `--yes` / `-y` | off | Non-interactive: accept all defaults |
| `--output` | `quant.config.yaml` | Output path |

> **Data scope → Factor Researcher.** `--lobster-level` and `--fundamentals` are
> written to `quant.config.yaml` and restrict the Factor Researcher to the fields
> the configured feed can actually serve: its brainstorm/codegen data-context lists
> only in-scope fields and any factor reading an out-of-scope field is dropped at
> persist. Default `lobster_level: 3` → existing configs are unchanged.

---

## Main Pipeline

### `run_fund.py`
Full end-to-end single pass: optional factor research → Selector → Architect → Statistician (×N) → Portfolio Manager.

| Flag | Default | Description |
|------|---------|-------------|
| `--research` | off | Run a Factor Researcher session first |
| `--n-strategies` | `3` | Number of strategy-research attempts |
| `--max-iterations` | `3` | Architect refinement iterations per strategy |
| `--oos-ratio` | `0.2` | Fraction of data held out for the Statistician |
| `--target-horizon` | `6` | Forecast horizon in bars (1=10s, 6=1min, 60=10min) |
| `--n-tickers` | all | Cap universe size |
| `--no-committee` | off | Single balanced PM instead of a 3-PM committee |
| `--voting` | `simple_average` | Committee aggregation (`simple_average` / `weighted_average` / `llm_moderator`) |
| `--fresh` | off | Ignore existing DBs and start from an empty book |
| `--config` | `quant.config.yaml` | Alternate config file to run the whole data layer under |
| `--config-name` | derived | Config scope name under `data/workspaces/<name>/` (default e.g. `yfinance_equity_demo`) |
| `--prerun <name>` | `base` | Prerun (research-LLM batch) within the config scope; `base` = seed alphas only |
| `--no-seeds` | off | Hide seed alphas, use only the prerun's researched factors |
| `--out-dir` | — | *[deprecated]* full override: write every DB + returns + models under this dir |

Default output: `data/workspaces/<config>/preruns/<prerun>/` (factors, strategies, returns, models, portfolio, showcase).

---

### `run_factor_research.py`
Runs the Factor Researcher agent only — mines factors from papers into a **(config, prerun) scope**. Never writes the seed-only main DB.

| Flag | Default | Description |
|------|---------|-------------|
| `--name <id>` | `base` | Prerun name → `data/workspaces/<config>/preruns/<name>/` |
| `--config-name` | derived | Config scope name (default from the active config) |
| `--model <llm>` | — | LLM model override for the researcher |
| `--llm-provider` | `openai` | LLM provider (`openai` / `anthropic` / …) |
| `--target-factors` | `100` | Stop after this many successful factors |
| `--papers-per-session` | `25` | Papers to read per session |
| `--ideas-per-session` | `25` | Factor ideas to brainstorm per session |
| `--max-sessions` | `12` | Max research sessions |
| `--horizon` | `6` | IC backtest horizon in bars |
| `--n-tickers` | `15` | Universe size for IC backtest |
| `--sample` | `random` | Paper sampling: `random` or `unread_first` |
| `--dedup-scope` | `package` | Dedup against `package` (global) or `prerun` (only this run) |
| `--data-dir` | `ticker_data` | Data directory |
| `--reset` | off | Clear this scope's factors/code/read-log before starting |

---

### `run_merge.py`
Composes an *active book* under `data/books/<name>/` by pooling selected scopes' researched factors **and** strategies (artifacts + returns copied, paths rewritten). The seed-only main library is never touched. Idempotent.

| Flag | Default | Description |
|------|---------|-------------|
| `--from <config/prerun> …` | — | One or more source scopes (mutually exclusive with `--from-config`) |
| `--from-config <config>` | — | Pool every prerun under this config |
| `--into <book>` | required | Destination book name (`data/books/<name>/`) |
| `--factors <ids…>` | all | Specific factor ids to merge |
| `--factors-only` | off | Merge factors only — don't pool strategies/artifacts |
| `--on-collision` | `skip` | `skip` (idempotent) or `overwrite` |
| `--dry-run` | off | Report only; write nothing |
| `--run-pm` | off | After merging, run the PM committee over the composed book |
| `--no-committee` | off | With `--run-pm`: single PM instead of a committee |

---

### `scripts/migrate_main_seed_only.py`
One-shot migration: make `data/factors/factor_db.json` seed-only by moving researcher factors into a preserved `legacy` scope (backed up, idempotent).

| Flag | Default | Description |
|------|---------|-------------|
| `--dry-run` | off | Report only; write nothing |
| `--no-backup` | off | Skip the timestamped backup of the main DB |
| `--migrate-preruns` | off | Also copy existing flat preruns into config-scoped workspaces |
| `--config-name` | derived | Config scope for `--migrate-preruns` |

---

### `run_model_comparison.py`
Compares factor sets from multiple research-LLM preruns on **four** tracks: single-factor IC, factor analytics (diversity/redundancy + deflation/importance), ML-combined-signal brute-force (catalog models + ensemble → per-underlying vectorised backtest), and the full downstream agent fund. Outputs figures, CSV/JSON tables, `report.md`, and `comparison.ipynb` under `data/comparisons/<id>/`; tables+figures are checkpointed after each track (`status.json`) so an interrupt never loses progress. Only the downstream track (and `--research`) spends LLM.

**Data source:** provider-aware — the shared panel follows `quant.config.yaml`. With `provider: lobster` it uses the direct LOBSTER CSV loader (the full microstructure panel) and `--data-dir` points at the per-ticker CSV root; with an API provider (`yfinance` / `fmp` / `alphavantage`) it loads through the abstracted data layer (universe / timespan from the config) and `--data-dir` is ignored.

| Flag | Default | Description |
|------|---------|-------------|
| `--preruns <a,b,…>` | — | Comma-separated prerun names to compare |
| `--all` | off | Compare all existing preruns |
| `--models <m1,m2,…>` | — | Filter to specific model types in the brute-force track |
| `--no-ensemble` | off | Skip the equal-weight ensemble in the brute-force track |
| `--include-seeds` | off | Include seed alphas alongside researched factors |
| `--no-ic` | off | Skip the single-factor IC track |
| `--no-analytics` | off | Skip the factor-analytics track (diversity + deflation/importance) |
| `--no-bruteforce` | off | Skip the brute-force ML track |
| `--no-downstream` | off | Skip the downstream agent track (the only LLM-spending step) |
| `--n-strategies` | `3` | Strategy attempts per prerun in the downstream track |
| `--horizon` | `6` | Forecast horizon in bars (brute-force + downstream) |
| `--oos-ratio` | `0.2` | OOS tail fraction — last N% of bars are OOS. **Ignored when `--train-months`/`--oos-months` are given** (calendar split takes over) for the LLM-free tracks; the downstream agent track always uses this ratio. |
| `--train-months <spec>` | — | Calendar IS/train window instead of the `--oos-ratio` tail. A comma list of months/dates (`2024-06,2024-07` / `2024-06-15`) or an inclusive range (`2024-06:2024-08`). **Requires `--oos-months`** and must be disjoint from it; the panel is restricted to train∪OOS so every track scores those months. |
| `--oos-months <spec>` | — | Calendar OOS window (same format as `--train-months`); must be disjoint from it. |
| `--n-tickers` | all | Cap universe size by **count** (smaller panel = faster every track) |
| `--tickers <a,b,…>` | all | Name the **exact** underlyings (e.g. `SPY,QQQ,CORN`); overrides `--n-tickers`. Missing names are logged, not fatal. |
| `--fast` | off | Fast preset: subsample training rows (`--train-sample-frac` 0.1) + slim panel (`--max-bars` 20000) + lighter tree/boost params |
| `--train-sample-frac` | `1.0` | Fraction (0–1] of training ROWS used to fit each model (0.1 under `--fast`) |
| `--max-bars` | all | Uniformly stride the panel to ≤ N timestamps (biggest speed lever; 20000 under `--fast`) |
| `--analytics-max-rows` | `50000` | Cap (timestamp×ticker) rows in the analytics correlation / importance fits |
| `--corr-threshold` | `0.7` | \|corr\| ≥ τ groups factors into a redundancy cluster |
| `--no-checkpoint` | off | Disable persisting tables/figures after each track |
| `--fit-scope` | `pooled` | Fit ONE model across all underlyings (`pooled`, suits homogeneous/data-light universes e.g. yfinance S&P100) or a SEPARATE model per underlying (`per_underlying`, suits heterogeneous/data-rich ones e.g. the LOBSTER ETFs) |
| `--fit-standardize` | `per_underlying` | Factor standardisation before the ML fit: `per_underlying` / `cross_sectional` |
| `--position-mode` | `threshold` | Map combined signal → position: `threshold` / `sign` / `continuous` |
| `--position-threshold` | `1.0` | ±t (z units) for the threshold band |
| `--position-zscore` | `expanding` | Per-underlying z-score basis: `expanding` / `full` / `rolling` / `none` |
| `--position-zscore-window` | `500` | Window for the `rolling` z-score basis |
| `--aggregation` | `portfolio` | `portfolio` (one netted book) or `per_underlying` (mean/std across names) |
| `--data-dir` | `ticker_data` | LOBSTER provider only (per-ticker CSV root); ignored for API providers |
| `--out-dir` | auto | Override output folder |
| `--research` | off | Mine preruns first before comparing |
| `--prerun-spec name=model[:provider]` | — | With `--research`: define preruns to mine (repeatable) |
| `--target-factors` | `50` | Target factors per prerun when `--research` is used |
| `--dedup-scope` | `prerun` | Dedup scope when `--research` is used |

---

## Individual Stage Scripts (`run_pipeline/`)

Run one stage at a time against the live databases — useful for debugging or re-running a single step.

### `run_pipeline/1_factor_research.py`
Factor Researcher only: reads papers, brainstorms ideas, generates factor code, runs IC backtest, persists survivors.

| Flag | Default | Description |
|------|---------|-------------|
| `--session-id` | today's date | Tag for this research session |
| `--cutoff` | — | ISO date; IC backtest uses only data before this date |
| `--n-papers` | `2` | Papers to read |
| `--n-ideas` | `3` | Factor ideas to brainstorm |
| `--horizon` | `6` | IC backtest horizon in bars |
| `--sample-strategy` | `unread_first` | Paper sampling: `unread_first` or `random` |
| `--reset` | off | Clear researcher factors before starting |

### `run_pipeline/2_selector.py`
Selector only: reads the factor catalog, formulates a trading hypothesis, and outputs the selected factor set. No arguments.

### `run_pipeline/3_architect.py`
Selector → Architect: runs the Selector to get a hypothesis + factor set, then runs the Architect fit + backtest refinement loop.

| Flag | Default | Description |
|------|---------|-------------|
| `--max-iterations` | `3` | Architect refinement iterations |
| `--oos-ratio` | `0.2` | OOS fraction (reserved for Statistician) |
| `--target-horizon` | `6` | Forecast horizon in bars |

### `run_pipeline/4_statistician.py`
Full strategy-research pass: Selector → Architect → Statistician, with optional persist.

| Flag | Default | Description |
|------|---------|-------------|
| `--max-iterations` | `3` | Architect refinement iterations |
| `--oos-ratio` | `0.2` | OOS fraction |
| `--target-horizon` | `6` | Forecast horizon in bars |
| `--no-persist` | off | Run the Statistician but don't write the strategy to the DB |
| `--fresh` | off | Ignore existing strategy DB, start clean |

### `run_pipeline/5_portfolio_manager.py`
Portfolio Manager over the existing strategy book.

| Flag | Default | Description |
|------|---------|-------------|
| `--pm-name` | — | Name tag for this PM run |
| `--mode` | `selector` | `selector` (rule-based) or `active` (LLM-assisted) |
| `--personality` | `balanced` | `defensive` / `balanced` / `aggressive` |
| `--committee` | — | Comma-separated personalities to run as a committee |
| `--voting` | `simple_average` | Committee aggregation: `simple_average` / `weighted_average` / `llm_moderator` |
| `--method` | — | Override construction method (e.g. `equal_weight`, `min_variance`, `hrp`, …) |
| `--target-n` | — | Override target book size |

### `run_pipeline/run_pipeline.py`
All stages in sequence with selective skips.

| Flag | Default | Description |
|------|---------|-------------|
| `--skip-research` | off | Skip the Factor Researcher stage |
| `--skip-strategy` | off | Skip Selector → Architect → Statistician |
| `--skip-pm` | off | Skip the Portfolio Manager |
| `--n-papers` | `2` | Papers for the researcher |
| `--n-ideas` | `3` | Ideas to brainstorm |
| `--horizon` | `6` | IC backtest horizon in bars |
| `--n-strategies` | `3` | Strategy attempts |
| `--max-iterations` | `3` | Architect iterations per strategy |
| `--oos-ratio` | `0.2` | OOS fraction |
| `--target-horizon` | `6` | Forecast horizon in bars |
| `--no-committee` | off | Single PM instead of committee |
| `--voting` | `simple_average` | Committee aggregation method |
| `--fresh` | off | Start from empty strategy/portfolio DBs |

---

### `run_backtest.py`
!!! not heavily tested, possibly bugs, especially with newer multi-provider features !!!!
Walk-forward backtest: runs weekly research/strategy/PM meetings over a date range and trades the resulting book with spread-aware costs.

| Flag | Default | Description |
|------|---------|-------------|
| `--start` | required | ISO start date |
| `--end` | required | ISO end date |
| `--warmup` | `1M` | Warm-up before first meeting |
| `--grid-freq` | `1W` | Rebalance-grid step |
| `--research-every` | `1W` | Factor research cadence |
| `--strategy-every` | `1W` | Strategy meeting cadence |
| `--pm-every` | `1W` | PM rebalance cadence |
| `--no-research` | off | Skip factor research meetings |
| `--factor-universe` | `all` | `all` (global DB) or `session` (only this run's factors) |
| `--prerun <name>` | — | Use a named prerun's factor set |
| `--no-seeds` | off | Hide seed alphas (with `--prerun`) |
| `--initial-strategies` | `5` | Strategies built at first meeting (bootstrap) |
| `--n-strategies` | `2` | Strategies attempted per meeting after the first |
| `--max-iterations` | `3` | Architect iterations per strategy |
| `--oos-ratio` | `0.2` | OOS fraction |
| `--target-horizon` | `6` | Forecast horizon in bars |
| `--execution` | `netted` | `netted` (one fund book, opposite signals net out) or `pod` (per-strategy books) |
| `--fund-max-positions` | `50` | Max positions in the consolidated book |
| `--max-weight-per-name` | `0.10` | Max weight per ticker |
| `--gross-leverage` | `1.0` | Gross leverage cap |
| `--commission-bps` | `0.5` | Fixed commission in bps |
| `--no-spread-cost` | off | Disable half-spread execution cost |
| `--no-committee` | off | Single PM instead of committee |
| `--n-tickers` | all | Cap universe size |
| `--data-dir` | `ticker_data` | Data directory |
| `--run-id` | auto | Name for the output folder under `data/backtests/` |
| `--resume` | off | Resume an existing run |

