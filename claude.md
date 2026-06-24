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
- **Architect** – combines factors into a strategy via a refinement loop. The
  fitted model maps (cross-sectionally normalised) factor signals → forward
  returns; its prediction is the composite signal, turned into positions by the
  configured **position-construction** regime (see the status note below).
- **Statistician** – OOS tests, deflated Sharpe, accept/reject gate.
- **Portfolio Manager** – screens, allocates capital, monitors/retires
  strategies (single PM or a committee).

State flows through shared databases (factor / paper / strategy / portfolio).
The stage sequence lives in `quant_fund_agent/pipeline.py` so the notebook,
scripts, and backtests all drive the agents identically.

## Status
A working **MVP** of the full pipeline exists. Each stage and agent is
intended to be advanced significantly in future work.

**Walk-forward backtest: prerun factor-injection + rich analytics (done).** The
walk-forward harness (`run_backtest.py` + `quant_fund_agent/simulation/`) now has
a second **factor source** besides LLM research: `--factor-source prerun_inject`
replaces the per-meeting LLM Factor-Researcher with a cheap, deterministic draw
from one or more **preruns** — `--inject-preruns a,b` (under `--inject-config`,
default the derived data-config scope e.g. `yfinance_equity_sp100`) pools their
RESEARCHER factors (union, dedup), shuffles once with the run seed, pre-filters to
those **computable on the live panel** (mirrors the comparison harness's gating),
and injects `--factors-per-meeting` (default 2) onto the run catalog on each
research-due meeting (seeds always available; the pool is exhausted once every
factor has been injected). Factor *code* already lives in the shared
`factors/researcher/` package so signals compute with no extra wiring; only the
catalog *records* are moved (`simulation/factor_injection.py`). The harness is now
**config-driven for API providers** — `run_backtest.py --config quant.config.<x>.yaml`
exports `QF_CONFIG_FILE` before the panel loads, so a yfinance/FMP universe + date
range (e.g. the static, survivorship-biased `sp100` list since 2016) drives the
whole run; `quant.config.sp100.yaml` is the ready-made yfinance S&P100 config.
**Analytics** are now presentation-ready: the execution layer tracks per-bar gross
exposure + names held, `BacktestResults` adds a NAV path / % invested / drawdown to
`equity.csv` + `fund_metrics.json`, and `simulation/report.py` renders a `report/`
folder of figures (NAV+drawdown, cumulative return, percent invested & names held,
rolling Sharpe, monthly returns, per-strategy attribution, catalog/strategy growth)
plus a `report.md` KPI table. All meeting cadences (`--research-every`,
`--strategy-every`, `--pm-every`, `--grid-freq`) are independent, so "monthly
everything" is `--grid-freq 1M --research-every 1M --strategy-every 1M --pm-every 1M`.
Tests: `tests/test_simulation_injection.py`.

**Research-LLM comparison (done; extended with analytics + speed/reliability).**
Named factor-research *preruns* (`run_factor_research.py --name <id> --model <llm>
[--llm-provider <p>] --dedup-scope prerun`) mine N factors with a chosen research
model into a self-contained `data/factors/preruns/<name>/`. `run_model_comparison.py`
then compares several preruns' factor sets on **four** axes — **single-factor IC**
(by default a **per-underlying time-series IC** — Spearman of a factor's value
vector vs the underlying's *own* forward-return vector, pooled across underlyings,
so it is well defined for a single ticker and has no cross-section;
`--fit-standardize cross_sectional` switches back to cross-sectional rank-IC),
**factor analytics** (LLM-free:
*diversity/redundancy* — signal correlation, effective # of independent factors via the
participation ratio, cluster count; and *deflation/importance* — best |IC| haircut for
the number of factors tried, plus LASSO/GBM feature importance & sparsity), **ML-combined
signal → per-underlying vectorised backtest** (each catalog model + ensemble combines the
factors into ONE predicted signal, fit on IS; that combined signal is backtested *not*
cross-sectionally but as a standalone directional bet per underlying, OOS;
`comparison/vector_backtest.py` + `bruteforce.py`). To avoid the Sharpe bias from
overlapping forward returns (a single `position × h-bar forward return` row overlaps
`h−1` bars with its neighbour → annualised return inflated ~`h`× and Sharpe ~`√h`×),
each bar's target is held as a **staggered "tranche" book** — the live position is the
rolling mean of the last `holding_period` targets (`1/h` capital layered in per bar) —
**marked to market on the 1-bar forward return** (`book[t] × forward_return(t→t+1)`),
which is the same non-overlapping convention the deployed `strategy_backtester` uses, so
research and deployment can't drift. `--holding-period` sets the tranche length
(default = `--horizon`; the forecast/IC horizon stays `--horizon` regardless). Also
**downstream agents** (the full Selector→Architect→Statistician→PM
fund, single-pass OOS) — emitting presentation-ready figures, CSV/JSON tables, a
`report.md` and a `comparison.ipynb` under `data/comparisons/<id>/`
(`quant_fund_agent/comparison/`). The backtest's modelling choices are all CLI args (both
sides implemented): `--fit-scope {pooled,per_underlying}` (default **pooled** — ONE model
across all underlyings; `per_underlying` fits a separate model per name, for heterogeneous
data-rich universes like the LOBSTER ETFs), `--position-mode {threshold,sign,continuous}`
(default threshold band), `--position-zscore {expanding,full,rolling,none}`,
`--aggregation {portfolio,per_underlying}`, `--fit-standardize {per_underlying,cross_sectional}`
(default per-underlying time-series z-score on IS stats). **`--fit-standardize` now governs
the WHOLE comparison**: at the default `per_underlying` the IC track, the analytics
diversity/importance fits (`comparison/analytics.py` `_feature_matrix`) and the brute-force
fit + combined-signal IC are *all* per-underlying (shared
`comparison/standardize.per_underlying_zscore`) — **no cross-section anywhere** — so every
track is meaningful for a single ticker; `cross_sectional` restores the legacy across-tickers
z-score + rank-IC. `--importance-top-n` (default 10) caps factors per (prerun, model) in the
importance table; set it high to keep the full per-factor vector. Everything except the
downstream track and `--research` is LLM-free.
**Universe + split selection:** beyond `--n-tickers N` (a count) you can name the exact
underlyings with `--tickers AAPL,MSFT,CORN` (overrides the count), and beyond `--oos-ratio`
(a tail fraction) you can split IS/OOS by the *calendar* with `--train-months`/`--oos-months`
— each a comma list of months/dates (`2024-06,2024-07` / `2024-06-15`) or an inclusive range
(`2024-06:2024-08`); they must be disjoint, the panel is restricted to their union so every
track scores those months, and the split seam is `ComparisonConfig.split_masks(index)` (used
by the brute-force/vector-backtest tracks; the LLM downstream track keeps its ratio split).
**Faster + crash-safe:** `--max-bars N` (default 20000 under `--fast`)
uniformly strides the panel so *every* track is fast and the brute-force OOM is gone;
the harness **checkpoints tables+figures after each track** (writing `status.json`),
so an interrupted run never loses completed tracks. Factors needing fields the current
data lacks are filtered (and reported), so it runs on today's LOBSTER sample now and
re-runs unchanged once full LOBSTER / FMP data is downloaded.
**Rolling-window sweep (`run_rolling_comparison.py` + `comparison/rolling.py`).** Automates
the comparison **per ticker over a rolling IS/OOS month window** (default 2 IS months + the
next OOS month, stepping one month forward so the prior OOS month becomes the second IS
month) for every ticker under `ticker_data/`, comparing the preruns **per underlying**. Each
(ticker, window) runs in its **own subprocess** — the OS reclaims its memory between runs, so
the large intraday panel never accumulates (no OOM); the sweep is **resumable** (windows whose
`status.json` is all-`ok` are skipped) and robust (a failed run is logged; the sweep
continues). It then **aggregates** every run into `data/comparisons/<batch>/`:
`combined/{bruteforce,importance,diversity,ic}_all.csv` (tagged `ticker,oos_month,is_window`),
per-ticker `importance_over_months__<prerun>__<model>.csv` + heatmaps (the **most important
features over the OOS months**) and `performance_<metric>__<model>.png`, plus `summary.md` and
`manifest.json`. Tests: `tests/test_rolling_comparison.py`.

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

**Position construction — configurable, default by data type (done).** How a
deployed strategy's composite signal becomes positions is now a regime, resolved
once per run and **stamped on the StrategySpec/StrategyRecord** so the Architect
IS-fit, the Statistician OOS test and live/PM all reproduce the same book:
**`cross_sectional`** (the prior behaviour — cross-sectionally rank the signal,
trade the top `max_positions`, scale to a dollar-neutral long/short book) vs
**`per_underlying`** (standardise each name over its *own* history, go
long/flat/short by a boundary — `position_mode` `threshold`/`sign`/`continuous`,
`position_zscore_basis` default causal `expanding` — then size each active name
to **`1/max_positions`** equal weight; directional, net market exposure, not
dollar-neutral). Default is **provider-keyed**: `per_underlying` for LOBSTER
(heterogeneous, data-rich ETFs), `cross_sectional` for the API/stock universes;
override with `run_fund.py --position-construction {auto,cross_sectional,per_underlying}`
or `QF_POSITION_CONSTRUCTION`. The two boundary primitives (`zscore_over_time`,
`directional_positions`) live in `backtesting/positions.py` and are **shared**
with the research comparison track (`comparison/vector_backtest.py`) so research
and deployment can't drift. The model catalog is unchanged (linear gives factor
weights; trees/boosting stay available). Equal-weight `1/max_positions` sizing is
the v1; inverse-vol / other sizing is a documented future extension. Tests:
`tests/test_position_construction.py`.

**Per-factor prediction horizons — done.** A factor no longer borrows a single
externally-passed horizon; it carries its own. `BaseFactor.prediction_horizon`
(canonical, validated like `inputs`) + optional `suggested_horizons` (in *bars*)
are stamped onto `FactorRecord` at persist (the dual of `required_inputs`). The
Factor Researcher now *chooses* the horizon: the brainstorm/codegen prompts get a
**bar-size-aware HORIZON CONTRACT** (seconds-per-bar inferred from the panel index
via `data/frequency._median_bar_seconds`, surfaced by `build_data_context(...,
seconds_per_bar)` — no hardcoded "10s"), and `codegen` requires a positive-int
`prediction_horizon ≤ MAX_PREDICTION_HORIZON`. Existing factors were backfilled to
**constant 6** (`FactorDatabase.backfill_prediction_horizons`,
`scripts/backfill_horizons.py`; idempotent via a `metadata` sentinel; a
`data_driven` best-`|IC-IR|` mode also exists). **IC** is now anchored at each
factor's *own* horizon: `engine.backtest_factor(..., factor_horizon=)` unions it
into the grid and headlines it (`factor_horizon=None` is byte-identical to the old
path), and the comparison IC track emits `ic_own`/`icir_own`/`horizon_own` +
`mean_abs_ic_own` alongside the shared grid. **Combined signal:** the brute-force
track derives its forecast horizon as the **mode** of the constituent factors'
horizons (`ComparisonConfig.resolve_target_horizon`, `combined_horizon_agg ∈
{mode,median,max,min,explicit}`; `run_model_comparison.py --horizon` is now an
*override* → `explicit`, `--combined-horizon-agg` sets the default); the
**downstream Architect** is shown each factor's `prediction_horizon` and **chooses**
`StrategySpec.target_horizon` (was config-fixed), persisting it across revise
iterations. Tests: `tests/test_factor_horizon_codegen.py`,
`tests/test_factor_horizon_backfill.py`, `tests/test_horizon_pipeline.py`,
extended `tests/test_comparison.py`.

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

**Survivorship-bias-free S&P 500 — point-in-time membership (done).** A static
ticker list over a 2010→today backtest over-represents survivors. `DataSettings.
membership="sp500"` (or `QF_MEMBERSHIP`) turns the universe **time-varying**:
`resolve_universe` returns the **union of every name ever an S&P 500 member** in
`[start,end]` (834 distinct tickers since 2010 vs ~503 today) so names that later
left still load, and `data/panel.load_panel` applies a **per-bar boolean mask**
once at load (`data/membership.py::apply_membership_mask`) — `NaN`-ing every
`(date,ticker)` cell where the ticker wasn't a constituent. Because it's the single
panel-load seam, research, the Architect/Statistician, the walk-forward trade loop
and the comparison harness are **all** survivorship-correct with no per-loop change
(cross-sectional ops skip the NaNs; the cutoff `_truncate_as_of` is orthogonal).
The canonical interval table `data/universes/membership/sp500.csv`
(`ticker,name,start_date,end_date,add_reason,remove_reason,source`; end exclusive)
is reconstructed from **free public sources** by `scripts/build_sp500_membership.py`
— **primary** GitHub `fja05680/sp500` (date→full-set series, forward run-length
scan into spells) cross-checked against **Wikipedia** (current + "Selected changes"
backward-walk, month-end Jaccard 0.91 mean rising 0.84→0.99; reasons + rename
detection). Renames (FB→META, …) coalesce into one continuous spell under the
current ticker. Audited (month-end count 497–506; no overlaps; TSLA absent-2015/
present-2022, ATVI present-2016/gone-2024) in-build and in `tests/test_membership.py`.
Build is idempotent + offline-replayable (dated raw snapshots under
`membership/sources/`, `MANIFEST.json`). **Free path is tickers-only** — yfinance
can't serve most delisted names (≈82% sampled coverage, the residual bias a premium
`CrspSource`/`FmpSource` closes via the `MembershipSource` seam). The static
`sp100.txt` is untouched. See `docs/data-layer/SP500_MEMBERSHIP.md`.

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
