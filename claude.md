# QuantFundAgent

Master's thesis project (Mathematics & Finance, Imperial College London).

## Goal
Build LLM agents that operate like a professional quant fund: researching
features, designing strategies, and deploying them live.

## Architecture
A LangGraph multi-agent pipeline mirroring a quant fund's org chart:

- **Factor Researcher** – reads papers, brainstorms alpha ideas, generates
  factor code, backtests each one (recording its IC for reference) and keeps
  every factor that runs — IC magnitude is not a keep/drop gate. It only ever
  invents factors the *configured data feed can serve*: its brainstorm/codegen
  data-context lists only in-scope fields and any factor reading an out-of-scope
  field is dropped at persist (see the data-scope status note below).
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

**Research-LLM comparison (done; extended with analytics + speed/reliability).**
Named factor-research *preruns* (`run_factor_research.py --name <id> --model <llm>
[--llm-provider <p>] --dedup-scope prerun`) mine N factors with a chosen research
model into a self-contained `data/factors/preruns/<name>/`. `run_model_comparison.py`
then compares several preruns' factor sets on **four** axes — **single-factor IC**
(cross-sectional rank-IC, raw quality), **factor analytics** (LLM-free:
*diversity/redundancy* — signal correlation, effective # of independent factors via the
participation ratio, cluster count; and *deflation/importance* — best |IC| haircut for
the number of factors tried, plus LASSO/GBM feature importance & sparsity), **ML-combined
signal → per-underlying vectorised backtest** (each catalog model + ensemble combines the
factors into ONE predicted signal, fit on IS; that combined signal is backtested *not*
cross-sectionally but as a standalone directional bet per underlying — `position(signal) ×
the underlying's own forward return`, OOS; `comparison/vector_backtest.py` +
`bruteforce.py`), and **downstream agents** (the full Selector→Architect→Statistician→PM
fund, single-pass OOS) — emitting presentation-ready figures, CSV/JSON tables, a
`report.md` and a `comparison.ipynb` under `data/comparisons/<id>/`
(`quant_fund_agent/comparison/`). The backtest's modelling choices are all CLI args (both
sides implemented): `--position-mode {threshold,sign,continuous}` (default threshold band),
`--position-zscore {expanding,full,rolling,none}`, `--aggregation {portfolio,per_underlying}`,
`--fit-standardize {per_underlying,cross_sectional}` (default per-underlying time-series
z-score on IS stats). Everything except the downstream track and `--research` is LLM-free. **Faster + crash-safe:** `--max-bars N` (default 20000 under `--fast`)
uniformly strides the panel so *every* track is fast and the brute-force OOM is gone;
the harness **checkpoints tables+figures after each track** (writing `status.json`),
so an interrupted run never loses completed tracks. Factors needing fields the current
data lacks are filtered (and reported), so it runs on today's LOBSTER sample now and
re-runs unchanged once full LOBSTER / FMP data is downloaded.

**Researcher data-scope gating — done.** The setup wizard now asks which data
the Factor Researcher may use, and that choice flows all the way through to the
factors it invents. For LOBSTER it asks the **order-book level** — `lobster_level`
in `quant.config.yaml`: **2** = order book only (mid-derived OHLC, traded volume,
spread/depth/imbalance + per-level book), **3** (default) adds the
message-stream fields that can't be reconstructed from the book (`trade`,
`orderFlow`, `hidden`, `auction`, `effSpread`, `trdLiq`, `ofLiq`,
`nbEvents/nbHidden/nbTrades`). For the API vendors it asks whether to use
**fundamental** data (the existing `fundamentals` flag; equity FMP/AlphaVantage
only). The split lives in `data/tiers.py` (`lobster_fields_for_level`,
`LOBSTER_L2/L3_MICRO_FIELDS`); `LobsterProvider.available_fields()` honors the
level so gating is consistent *everywhere* (seed factors, comparison harness,
catalog), with **level 3 the default → existing configs unchanged**.
`data.usable_fields(settings)` is the single run-scope field set (provider caps −
non-OHLCV when fundamentals off); `pipeline.run_research_session` computes it and
threads it onto `FactorResearcherState.allowed_fields`. The brainstorm/codegen
DATA CONTEXT is built per-run by `agents/factor_research/prompts.build_data_context`
(only in-scope fields, empty sections dropped, LOBSTER-vs-generic intro), and
`filter_and_persist` drops any factor whose declared `inputs` fall outside the
scope. Tests: `tests/test_factor_research_data_scope.py`.

**Pluggable data layer (Phases 0–6 done — milestone complete).** The fund no
longer requires the author's LOBSTER CSVs: any panel load goes through
`quant_fund_agent/data/` (provider abstraction + parquet cache + capability
gating + calendar-aware annualization). Providers: `lobster` (local CSVs),
`yfinance` (key-free daily), `fmp`, `alphavantage` — the three API vendors are
**multi-asset (equity / crypto / fx)** via canonical `BASE-QUOTE` symbols
(`data/symbols.py`); crypto annualizes at 365 (inferred from weekend bars). Run
`python -m quant_fund_agent.setup` to pick a provider/asset-class/universe/
timespan and write `quant.config.yaml`, or
`python -m quant_fund_agent.setup --assist "<plain-English description>"` to have
an LLM draft a config you confirm (`setup_assist.py`; validated to legal values,
falls back to the deterministic wizard with no LLM key). See
`docs/data-layer/ROADMAP.md`.

**Non-OHLCV data — Stage 7 done (fundamentals + estimates/events).** Factors can
now read availability-stamped fundamentals (`sector`/`peRatio`/`roe`/`revenue`/…),
analyst estimates (`epsEstimate`/`revenueEstimate`) and earnings events
(`epsSurprise`) from FMP + AlphaVantage. Point-in-time is enforced in the data
layer (`data/fundamentals.py`: stamp at filing/`reportedDate` else fiscal-end +
lag → forward-fill onto the daily index with a staleness cap; `_truncate_as_of`
then keeps look-ahead out). Canonical field vocabulary + per-vendor normalization
(`data/fields.py`), enriched `fundamental` tier + new `estimates`/`events` tiers
(gating unchanged), a quarterly-TTL record cache (`cache.py::cached_records`), an
additive `ApiProvider._fetch_fundamentals` hook, the `indneutralize` wide-frame
fix, and example factors in `factors/fundamentals/`. Equity-only; `QF_FUNDAMENTALS=0`
to opt out. Live: AV's free tier delivers these; FMP's free tier serves only
`profile` and paywalls the rest (factors degrade, never crash). **Next (designed,
not built):** sentiment + macro — see `docs/data-layer/FUNDAMENTAL_AND_ALT_DATA.md`.

**Strict modularisation by (config, prerun) — done.** Researched factors and the
strategies built from them are isolated per **(data config, research-LLM prerun)**
under `data/workspaces/<config>/preruns/<prerun>/` (factors, strategies, returns,
fitted model artifacts, portfolio, showcase) via a single layout seam
(`quant_fund_agent/workspace.py`: `Scope`/`Book`, consumed by `run_fund.py`,
`run_factor_research.py`, the simulator and the merge tool). Returns and `.joblib`
artifacts are now scoped through the `STRATEGY_RETURNS_DIR` / `MODEL_ARTIFACT_DIR`
env seam (read live, inherited across the MCP subprocess — same pattern as
`FACTOR_DB_PATH`), so parallel runs never collide. The canonical *main* factor
library `data/factors/factor_db.json` holds **only seed/formulaic alphas** and is
never written by research; every factor/strategy is stamped with a `provenance`
(config hash + scope). `run_merge.py` (`quant_fund_agent/merge.py`) composes a
separate *active book* under `data/books/<name>/` by pooling chosen scopes'
factors + strategies for the PM — main is never mutated, cross-config pooling is
warned not blocked. `scripts/migrate_main_seed_only.py` splits the legacy mixed
main DB into seed-only main + a preserved `legacy` scope (backed up, idempotent,
`--dry-run`). `factors/preruns.py` is now a compatibility shim over `Scope`.

**LOBSTER ingestion — built & validated against the live site.**
`quant_fund_agent/data/lobster_ingest/` converts LOBSTER exports into
`ticker_data/{TICKER}/bin{YYYYMM}.csv` (`converter.py` + `scripts/convert_lobster.py`):
levels-agnostic, streams **one day at a time** (RAM-bounded), clips to the regular
session → 2340 bars/day, and **auto-detects the product** — *raw* message+orderbook
(19 aggregate cols) vs *sampled* 10s book (book cols only, flow cols empty). **Both
also emit per-level book columns** `askPrice{i}`/`askDepth{i}`/`bidPrice{i}`/
`bidDepth{i}` (`4·NumLevels` of them, bar-mean of price + displayed depth at each
level; `converter.output_columns(N)`/`level_field_names(N)`). The loader discovers
them from the CSV header and forwards each as its own panel field; the
microstructure tier advertises levels 1–10 so factors using them aren't gated out.
Derived columns are **reconstructed** from LOBSTER docs (original 2019 aggregator
isn't in the repo; defs + drift-warning in `docs/lobster-ingestion/CONVERSION_SPEC.md`).
`converter.convert_archive` streams a `.7z` day-by-day so a 2-yr raw archive never
unzips in full. The autonomous pull (`orchestrator.py` + `lobster_portal.py` +
`scripts/run_lobster_ingest.py`) drives the real portal: one-time `--recon` login
(persisted cookie `storageState`), `--place-raw` (single `requestdata.php` form,
one request/ticker), then `--ingest` reads finished orders off `mydata.php`,
stream-downloads each `.7z`, converts, and **deletes it** — resumable via
`orders_done.json`, disk-governed. Live-verified: downloaded+converted a CORN
level-10 order (500 days); a full **level-5 CORN pull (16 archives, 499 trading
days, 2024-06→2026-05, ~1.16M bars)** is converted into `ticker_data/CORN/` with
the 19 aggregate + 20 per-level columns (3 fully-empty thin-ETF days → `_empty_day`
NaN grids). **Key product finding:** only the **single `requestdata.php` form
yields RAW message+orderbook (all 19 columns)**; the **bulk
`requestBulkData.php` form delivers an order book only (no message stream → no
flow/trade columns)** regardless of `interval`. Download names: `…_LEVEL.7z` (raw)
vs `…_LEVEL_INTERVAL.7z` (sampled). Tests:
`tests/test_lobster_{converter,orchestrator}.py`. Scratch (`data/lobster_raw/`,
cookie `.lobster_state.json`) is gitignored. See `docs/lobster-ingestion/`.

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
