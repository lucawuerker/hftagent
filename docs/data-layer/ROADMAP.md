# Data-Layer Milestone — Roadmap (living doc)

> **For future agent sessions.** This is the master tracking doc for the
> "open-source data-vendor abstraction + guided onboarding" milestone. Update
> the checkboxes as phases land. Companion docs: [`ARCHITECTURE.md`](ARCHITECTURE.md),
> [`DATA_PROVIDERS.md`](DATA_PROVIDERS.md), [`ONBOARDING.md`](ONBOARDING.md).

## Why this milestone exists

QuantFundAgent currently runs on **one data shape only**: 10-second LOBSTER
microstructure CSVs that the author downloaded locally and cannot redistribute.
The thesis goal is to make this a clonable open-source project where **anyone
plugs in their own market-data API key (yfinance / FMP / AlphaVantage / …), runs
a guided setup to pick a universe + timespan + frequency, and drives the full
agent pipeline** — no LOBSTER data, no code edits.

## Locked design decisions

- **Capability tiers + synthesized derived fields.** Named field groups —
  `standard` (OHLCV), `fundamental` (+ sector/industry/subindustry/cap),
  `microstructure` (+ effSpread/spread/orderFlow/hidden/depth/lobImb/…). Each
  provider declares the fields it supplies; each factor's `inputs` resolve to a
  required tier; factors needing more than the active provider supplies are
  **gated out**, never crashed. `vwap` and `returns` are **auto-synthesized**
  from OHLCV, so they count as `standard`.
- **Multi-asset is the end goal, phased.** Core is built asset-class/calendar
  agnostic; **equities-daily ships first**; crypto/FX (24/7, 365-day
  annualization) is a later sub-step (Phase 6).
- **LLM-provider portability deferred** (stay on OpenAI this milestone).
- **Onboarding = deterministic CLI wizard + optional `--assist` LLM layer**,
  writing `quant.config.yaml`.

## Phases / stages

Each phase is independently shippable with its own test gate
(`./venv/bin/pytest` green after each).

- [x] **Phase 0 — Foundations (pure refactor, zero behavior change)** ✅ 2026-06-10
  - Central `quant_fund_agent/config.py` `Settings` (yaml + `.env`, env overrides).
  - `data/providers/base.py` `DataProvider` ABC; `LobsterProvider` wraps the
    existing CSV loader (impl stays in `backtesting/data_loader.py`).
  - `data/panel.py::load_panel(...)` dispatcher + `data/tiers.py`; routed the 4
    call sites (modeling/service, mcp/research_service, architect/graph;
    simulation/signals via modeling.service).
  - Gate met: 49 existing tests green; `tests/test_data_layer.py` proves the
    routed `load_panel` is byte-identical to the legacy loader.
- [x] **Phase 1 — Frequency-aware annualization** ✅ 2026-06-10
  - `data/frequency.py`: `bars_per_day_from_index` / `periods_per_year_from_index`
    via **median bar-spacing + session length** (robust to overnight gaps and to
    synthetic continuous panels; reproduces 2340 exactly for any 10-sec equity
    data). `DEFAULT_BARS_PER_DAY` / `TRADING_DAYS_PER_YEAR` are the single source
    of truth + fallback.
  - Data-driven at every consumer: `_compute_metrics` (per-strategy IS/OOS
    metrics), `simulation/results.fund_metrics` (+ turnover scaling), the
    statistician context builder (`mcp/statistics_service._build_context` infers
    from `data_full`, shared by MCP + in-process so parity holds), DSR test, and
    `pipeline.run_pm_rebalance` (infers covariance annualisation from the book's
    returns via new `_infer_annualisation`).
  - Gate met: full suite green; new tests assert 10-sec→2340/589680,
    daily→1/252, 1-min→390, crypto-daily→365.
- [x] **Phase 2 — Capability tiers + factor gating + derived fields** ✅ 2026-06-10
  - `data/tiers.py`: added `compatible_factors()` + `resolve_required_inputs()`
    (stored → registry → `["close"]`).
  - `FactorRecord` gained `required_inputs` + `required_tier` (`schemas.py`);
    populated on researcher creation (`factor_research/graph.py`) and backfilled
    onto the committed DB via `FactorDatabase.annotate_tiers()` +
    `scripts/annotate_factor_tiers.py` (197/197 → 98 standard, 99 microstructure).
  - **Gate at catalog READ time** (`mcp/catalog_service.load_factor_catalog`):
    filters by the active provider's `available_fields()`, annotates each row
    with `required_tier`, honours `QF_FACTOR_GATING=0`. Research backtests
    (`mcp/research_service.backtest_factors`) skip incompatible factors with a
    clean "gated: requires <tier> tier" reason.
  - `data/panel.py` synthesises `vwap=(H+L+C)/3` and `returns=close.pct_change()`
    (strip-from-request → load deps → derive → trim).
  - Gate met: full suite green (incl. MCP/in-process catalog parity); three
    executable verify scripts under `scripts/verify/` prove **no-op on LOBSTER
    (197/197)**, **98/197 on a standard provider** (99 microstructure gated),
    escape hatch, synthesis formulas, and the 4 known misdeclared alphas.
- [x] **Phase 3 — yfinance provider + onboarding MVP (first "clone-and-run")** ✅ 2026-06-10
  - `data/providers/yfinance.py` (no key, daily OHLCV, `auto_adjust=True`; network
    isolated to `_fetch` so the rest is testable offline). Registered in `PROVIDERS`.
  - `data/cache.py` per-symbol parquet cache (refetch-on-miss, atomic writes);
    `data/universe.py` + `data/universes/{demo,sp100}.txt` presets.
  - `quant_fund_agent/setup.py` wizard (`python -m quant_fund_agent.setup`):
    provider/key detection, prompts **and** CLI flags, validation fetch, writes
    `quant.config.yaml`.
  - Cost: opt-in flat `fallback_spread_bps` in `BacktestConfig` (default 0 =
    commission-only) — deliberately NOT a high-low proxy (overcharges ~100×).
  - Gate met: full suite green (69); offline verify (cache/universe/synthesis/
    252-yr/gating→98) + free live yfinance fetch + wizard round-trip, and a **real
    paid `run_fund.py --n-strategies 1` on yfinance with no LOBSTER data** ran
    Selector→Architect(ridge→lasso→xgboost)→Statistician(rejected, legit)→PM.
- [x] **Phase 4 — FMP + AlphaVantage providers** ✅ 2026-06-10
  - `data/providers/_http.py` shared throttle/retry GET; `fmp.py`
    (FMP **stable** API `historical-price-eod/dividend-adjusted` → adjusted OHLC)
    + `alphavantage.py` (`TIME_SERIES_DAILY`, **free tier = unadjusted + compact
    ~100 bars + ~25/day**, 13s throttle). Both `standard`-tier; registered.
  - Refactored the shared load flow into `base.py::ApiProvider` (yfinance/FMP/AV
    all extend it; only `available_fields` + `_fetch` differ).
  - **Fundamental tier deferred** — the 4 sector/cap factors pass a DataFrame to
    `indneutralize` (which wants a Series); they already work via graceful
    fallback. Tier + factor-shape fix = follow-up.
  - Gate met: full suite green (73); offline reshape/limit/load tests; **live FMP
    + AV fetches** (prices cross-check consistent); a **real paid `run_fund.py` on
    FMP** ran the full pipeline on adjusted daily data with no LOBSTER data.
- [x] **Phase 5 — Onboarding polish + docs** ✅ 2026-06-11
  - `quant_fund_agent/setup_assist.py` + `setup.py --assist`: an LLM turns a
    plain-English description into a **proposed** config. The proposal is fed to
    the deterministic wizard as *confirmable defaults* (precedence stays
    **CLI flag > LLM proposal > built-in default**); every value is validated
    against ground truth (usable providers, existing presets, coherent
    `start < end` dates) so the model can only narrow to legal values. Any assist
    failure (no LLM key / network / bad JSON) falls back to the deterministic
    wizard — onboarding never depends on an LLM. Built via the shared
    `llm.make_chat_llm` factory (`SETUP_ASSIST_LLM_MODEL` to override).
  - Fixed onboarding-doc drift (advertised `sp500`/`nasdaq100` presets that don't
    exist → actual `demo`/`sp100`, with a note on adding your own).
  - Gate met: full suite green (84); 11 offline assist unit tests
    (`tests/test_setup_assist.py`) cover validation/parsing/fallback/precedence;
    a **live** `scripts/verify/verify_phase5_assist.py` proved real LLM proposals
    stay in-bounds (incl. an adversarial "Bloomberg/weekly" prompt that was
    dropped/coerced), wrote a real config from free text, and confirmed graceful
    fallback under a forced LLM outage.
- [x] **Phase 6 — Multi-asset (crypto / FX)** ✅ 2026-06-11
  - **Calendar-aware annualization.** The Phase-1 `crypto→365` branch was dead
    code (no call site passed `asset_class`). Fixed *data-drivenly*:
    `frequency.py` infers the calendar from the index — weekend bars present ⇒
    365, else 252 (`_is_continuous_calendar` + `trading_days_per_year_from_index`),
    so every existing call site (`results.py`, `strategy_backtester.py`,
    `pipeline.py`, the statistician's DSR context via a new
    `StatTestContext.trading_days_per_year`) gets the right factor with no
    plumbing. Explicit `asset_class` still overrides (keeps the Phase-1 test).
  - **Multi-asset across all three API vendors.** `data/symbols.py` defines
    canonical `BASE-QUOTE` pairs and translates to each vendor's native form;
    each provider's `_fetch` dispatches on `asset_class` and returns frames keyed
    by the canonical symbol (cache/panel stay vendor-agnostic). yfinance
    (`BTC-USD`/`EURUSD=X`), FMP (`historical-price-eod/full`, raw OHLCV for
    crypto/fx), AlphaVantage (`DIGITAL_CURRENCY_DAILY`/`FX_DAILY`). All three now
    declare `asset_classes=("equity","crypto","fx")`; `lobster` stays equity-only.
  - **Validation + presets + wizard.** `panel.get_provider` rejects an
    unsupported provider/asset-class combo with a clear error; `crypto_demo` +
    `fx_demo` presets; the wizard + `--assist` are asset-class-aware (default the
    matching preset, clamp to the provider's classes). FX has no reliable volume
    (filled NaN; documented).
  - Gate met: full suite green (90); offline tests (weekend inference 365/252,
    symbol translation, per-vendor crypto/fx reshape incl. legacy AV `(USD)`
    format, asset-class validation). **Live** `verify_phase6_multiasset.py`:
    yfinance+FMP+AV crypto/fx fetched through `load_panel`, crypto→365 vs
    equity→252 proven, BTC-USD consistent to the dollar across all three vendors.
    A **paid `run_fund.py` on a 5-coin crypto universe** ran
    Selector→Architect(lasso/ridge)→Statistician(REJECT, legit)→PM on 730 daily
    crypto bars, 365-annualized, 98 factors visible (microstructure gated).

## Follow-ups (out of scope here)
- LLM-provider abstraction (OpenAI → +Anthropic/local) behind one env var.
- Live index-constituent resolution (point-in-time, survivorship-aware).

## Status log
- 2026-06-09: Plan approved; docs created; Phase 0 started.
- 2026-06-10: Phase 0 complete (data layer + routing, byte-identical, tests
  green). Phase 1 complete (frequency-aware annualization, full suite green).
- 2026-06-10: Phase 2 complete (capability gating + tier metadata + vwap/returns
  synthesis). Gating verified no-op on LOBSTER, 98/197 on a standard provider.
  `data/factors/factor_db.json` now carries `required_inputs`/`required_tier`.
- 2026-06-10: Phase 3 complete (yfinance provider + parquet cache + universe
  presets + onboarding wizard). Verified a real paid `run_fund.py` end-to-end on
  yfinance daily data with NO LOBSTER data. `quant.config.yaml` and `data/market/`
  are gitignored.
- 2026-06-10: Phase 4 complete (FMP + AlphaVantage providers; `ApiProvider` base).
  FMP uses the stable adjusted endpoint; AV free tier is unadjusted/compact/25-day
  (documented). Verified with live fetches + a paid `run_fund.py` on FMP. Three
  vendors now run the fund (yfinance/FMP/AV) + LOBSTER. Next: Phase 5 (wizard
  `--assist` + README/CLAUDE polish).
- 2026-06-11: Phase 5 complete (LLM-assisted onboarding `setup.py --assist` +
  `setup_assist.py`; doc-drift fix). Proposal validated/clamped to legal values,
  fed as confirmable defaults, with hard fallback to the deterministic wizard.
  Verified live (real proposals in-bounds incl. adversarial prompt; config
  written from free text; graceful fallback). Full suite green (84). Next: Phase 6
  (multi-asset crypto/FX).
- 2026-06-11: Phase 6 complete — **the milestone is done.** Crypto + FX on all
  three API vendors (yfinance/FMP/AV) via canonical-pair symbols + per-vendor
  translation; calendar-aware annualization (weekend bars ⇒ 365) fixes the dead
  crypto branch with zero plumbing; asset-class validation + crypto/fx presets +
  asset-class-aware wizard/assist. Verified live (crypto→365 vs equity→252;
  BTC-USD consistent across vendors) + a paid crypto `run_fund.py`. Full suite
  green (90). A forward-looking design doc for the *next* stage (non-OHLCV
  fundamental/alternative data fields) lives at
  [`FUNDAMENTAL_AND_ALT_DATA.md`](FUNDAMENTAL_AND_ALT_DATA.md).
  **Finding:** FX weekend-bar convention is vendor-dependent (FMP stamps weekend
  FX bars ⇒ 365; yfinance/AV are weekday-only ⇒ 252) — the inference correctly
  annualizes each series by its actual sampling rate.
