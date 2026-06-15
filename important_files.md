# Important Runnable Files

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
| `--cache-dir` | `data/market` | Parquet cache location |
| `--assist "<text>"` | — | LLM drafts a config from plain English |
| `--no-validate` | off | Skip validation fetch |
| `--yes` / `-y` | off | Non-interactive: accept all defaults |
| `--output` | `quant.config.yaml` | Output path |

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
| `--prerun <name>` | — | Use a named prerun's factor set instead of the global library |
| `--no-seeds` | off | With `--prerun`: hide seed alphas, use only researched factors |
| `--out-dir` | auto | Output directory for DBs + showcase |

---

### `run_factor_research.py`
Runs the Factor Researcher agent only — mines factors from papers into a named prerun or the global DB.

| Flag | Default | Description |
|------|---------|-------------|
| `--name <id>` | — | Prerun name (saves to `data/factors/preruns/<name>/`) |
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
| `--reset` | off | Clear existing prerun before starting |

---

### `run_model_comparison.py`
Compares factor sets from multiple research-LLM preruns on three axes: single-factor IC, brute-force ML, and the full downstream agent fund. Outputs figures, CSV/JSON tables, `report.md`, and `comparison.ipynb` under `data/comparisons/<id>/`.

| Flag | Default | Description |
|------|---------|-------------|
| `--preruns <a,b,…>` | — | Comma-separated prerun names to compare |
| `--all` | off | Compare all existing preruns |
| `--models <m1,m2,…>` | — | Filter to specific model types in brute-force track |
| `--no-ensemble` | off | Skip ensemble model in brute-force track |
| `--include-seeds` | off | Include seed alphas alongside researched factors |
| `--no-ic` | off | Skip single-factor IC track |
| `--no-bruteforce` | off | Skip brute-force ML track |
| `--no-downstream` | off | Skip downstream agent track (the only LLM-spending step) |
| `--n-strategies` | `3` | Strategy attempts per prerun in downstream track |
| `--horizon` | `6` | Forecast horizon in bars |
| `--oos-ratio` | `0.2` | OOS fraction |
| `--n-tickers` | all | Cap universe size |
| `--fast` | off | Faster brute-force (fewer CV folds) |
| `--train-sample-frac` | — | Subsample training data for speed |
| `--data-dir` | `ticker_data` | Data directory |
| `--out-dir` | auto | Override output folder |
| `--research` | off | Mine preruns first before comparing |
| `--prerun-spec name=model[:provider]` | — | With `--research`: define preruns to mine |
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

