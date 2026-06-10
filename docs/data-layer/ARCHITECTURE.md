# Data-Layer Architecture

> Companion to [`ROADMAP.md`](ROADMAP.md). This describes the *target* design of
> the pluggable data layer. Sections tagged **(planned)** are not built yet —
> check the roadmap checkboxes for current status.

## The problem being solved

Every quantitative computation in the system reads from **one ingestion seam**:

```python
load_panel(data_dir, tickers=None, fields=None, n_tickers=None, dtype="float32")
    -> dict[field_name -> pd.DataFrame(index=DatetimeIndex, columns=tickers)]
```

Historically this seam *was* the LOBSTER CSV loader
(`backtesting/data_loader.py`). To support arbitrary vendors we keep the **dict
contract** exactly and slide a provider abstraction underneath it.

The dict contract (unchanged):
- one key per **field** (`open`, `high`, `low`, `close`, `volume`, `vwap`,
  `returns`, plus microstructure fields when available);
- each value is a wide DataFrame, `DatetimeIndex` × tickers, aligned on a shared
  index;
- factors read `data["close"]` etc. and return a same-shaped signal frame.

## Components

```
quant_fund_agent/
├── config.py                 # Settings: quant.config.yaml + .env + env overrides
└── data/
    ├── providers/
    │   ├── base.py           # DataProvider ABC
    │   ├── lobster.py        # the original CSV logic, behind the ABC
    │   ├── yfinance.py       # (planned)
    │   ├── fmp.py            # (planned)
    │   └── alphavantage.py   # (planned)
    ├── tiers.py              # capability tiers (field groups) + helpers
    ├── frequency.py          # infer periods_per_year / bars_per_day from index
    ├── cache.py             # parquet cache keyed (provider, symbol, freq, asset)
    ├── universe.py           # universe presets + resolution
    └── panel.py             # load_panel(settings) → unified entry + PanelMeta
```

### `DataProvider` ABC (`data/providers/base.py`)

```python
class DataProvider(ABC):
    name: str
    asset_classes: tuple[str, ...]            # ("equity",), ("equity","crypto"), …

    def available_fields(self) -> set[str]:    # union of tiers it can supply
        ...

    def fetch(self, symbols: list[str], start, end, freq: str
              ) -> dict[str, pd.DataFrame]:    # field -> wide DataFrame
        ...
```

A provider is responsible only for **producing raw fields**. It does *not* know
about factors, annualization, or caching — those are orchestrated above it.

### `load_panel(settings)` (`data/panel.py`)

The single entry the rest of the system calls. Responsibilities:
1. Resolve the provider from `settings.data.provider`.
2. Consult the **parquet cache** (`data/cache.py`); `fetch()` only missing ranges.
3. Assemble the `dict[field -> DataFrame]`.
4. **Synthesize derived fields** when on the `standard` tier and absent:
   `vwap ≈ (high + low + close) / 3`, `returns = close.pct_change()`.
5. Return `(panel, PanelMeta)`.

`PanelMeta` carries the cross-cutting facts the rest of the system needs:
`periods_per_year`, `bars_per_day`, `available_fields`, `asset_class`, `freq`.
This is how frequency-aware annualization stops being hardcoded.

> **Look-ahead invariant.** The panel is loaded **full** and sliced by
> `as_of`/`cutoff` downstream (`modeling/service._truncate_as_of`). Caching
> changes *fetching*, never *slicing* — preserve this when touching the cache.

## Capability tiers (`data/tiers.py`)

Tiers are **named sets of fields**, not a strict linear order (a vendor can have
microstructure without fundamentals):

| Tier | Fields (cumulative over `standard`) |
|------|--------------------------------------|
| `standard` | `open, high, low, close, volume` (+ synthesized `vwap`, `returns`) |
| `fundamental` | `+ sector, industry, subindustry, cap` |
| `microstructure` | `+ effSpread, spread, orderFlow, hidden, depth, lobImb, effLobImb, nbTrades, …` |

- A provider's `available_fields()` is the union of tiers it supports.
- A factor is **compatible iff `set(factor.inputs) ⊆ provider.available_fields`**.
  (`vwap`/`returns` are always satisfiable because they're synthesized.)
- `required_tier(inputs)` returns a human-readable label (highest tier the
  factor touches), persisted on `FactorRecord` for display/filtering.

This is how the LOBSTER-specific factors (the `microstructure/` category) and
the classification factors quietly drop out when someone runs on plain daily
OHLCV, while ~95% of the seed library keeps working.

## Frequency-aware annualization (`data/frequency.py`)

The old code hardcoded 10-second bars: `BARS_PER_YEAR = 2340 * 252`,
`bars_per_day = 2340`. These are replaced by values inferred from the panel's
`DatetimeIndex` median spacing and carried on `PanelMeta`:

| Data | bars/day | periods/year |
|------|----------|--------------|
| 10s LOBSTER | 2340 | 2340 × 252 |
| 1-minute | 390 | 390 × 252 |
| daily equity | 1 | 252 |
| daily crypto (Phase 6) | 1 | 365 |

Threaded into: `pipeline.BARS_PER_YEAR`, the statistics test context
`bars_per_day` (`statistics/base.py`), the DSR test
(`statistics/tests/deflated_sharpe.py`), and the simulation's annualization.

## Transaction costs on vendors without a spread field

`simulation/execution.py::cost_rate_panel()` already **falls back to
commission-only** when `cfg.spread_field` is absent from the panel, so daily
data never crashes. Phase 3 *optionally* enriches this with a high-low synthetic
half-spread; the configurable `commission_bps` remains the floor.

## Configuration (`config.py` + `quant.config.yaml`)

A central `Settings` object replaces scattered import-time `os.getenv` calls.
Secrets stay in `.env`; everything else lives in `quant.config.yaml` (written by
the onboarding wizard). Env vars still override for back-compat, and the LOBSTER
defaults (`DATA_DIR=ticker_data`) are preserved. See [`ONBOARDING.md`](ONBOARDING.md)
for the config schema.
