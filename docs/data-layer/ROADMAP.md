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
- [ ] **Phase 2 — Capability tiers + factor gating + derived fields**
  - `data/tiers.py`; `compatible_factors(provider_fields)`.
  - Gate factor universe at: factor-DB build (`run_all_factors.py`), Selector
    catalog (`mcp/catalog_*`), research backtests (`mcp/research_service.py`).
  - Add `required_inputs` + `required_tier` to `FactorRecord` (`schemas.py`).
  - Synthesize `vwap`/`returns` in `data/panel.py`.
- [ ] **Phase 3 — yfinance provider + onboarding MVP (first "clone-and-run")**
  - `data/providers/yfinance.py` (no key, daily OHLCV, adjusted close).
  - `data/cache.py` parquet cache; `data/universe.py` + presets in `data/universes/`.
  - `setup_wizard.py` (`python -m quant_fund_agent.setup`): key detection,
    prompts, validation fetch, write `quant.config.yaml`.
  - Optional: high-low synthetic-spread cost enrichment in `simulation/execution.py`.
  - Gate: wizard + `run_fund.py` run end-to-end on yfinance with no `ticker_data/`.
- [ ] **Phase 4 — FMP + AlphaVantage providers**
  - `data/providers/fmp.py`, `alphavantage.py` (requests, key from `.env`,
    rate-limit-aware cache). These can fill the `fundamental` tier.
- [ ] **Phase 5 — Onboarding polish + docs**
  - `setup_wizard.py --assist` (LLM proposes config, user confirms).
  - Finalize these docs; update `README.md` + `CLAUDE.md`.
- [ ] **Phase 6 — Multi-asset (LATER)**
  - Asset-class-aware calendars + 365-day annualization; crypto/FX providers.
    Enabled by `asset_class` already threaded through `PanelMeta`/`Settings`.

## Follow-ups (out of scope here)
- LLM-provider abstraction (OpenAI → +Anthropic/local) behind one env var.
- Live index-constituent resolution (point-in-time, survivorship-aware).

## Status log
- 2026-06-09: Plan approved; docs created; Phase 0 started.
- 2026-06-10: Phase 0 complete (data layer + routing, byte-identical, tests
  green). Phase 1 complete (frequency-aware annualization, full suite green).
  Next: Phase 2 (capability tiers + factor gating + `FactorRecord.required_tier`
  + vwap/returns synthesis).
