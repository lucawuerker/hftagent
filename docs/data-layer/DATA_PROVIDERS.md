# Adding a Data Provider

> Companion to [`ARCHITECTURE.md`](ARCHITECTURE.md). Step-by-step guide for
> wiring a new market-data vendor into QuantFundAgent.

## The contract

A provider subclasses `quant_fund_agent/data/providers/base.py::DataProvider`
and implements two things:

```python
from quant_fund_agent.data.providers.base import DataProvider

class MyVendorProvider(DataProvider):
    name = "myvendor"
    asset_classes = ("equity",)

    def available_fields(self) -> set[str]:
        # Which raw fields can you supply? Drives factor gating.
        # Use the tier helpers so you don't hand-maintain field lists:
        from quant_fund_agent.data.tiers import TIERS
        return set(TIERS["standard"])          # OHLCV-only vendor

    def fetch(self, symbols, start, end, freq) -> dict[str, pd.DataFrame]:
        # Return {field -> wide DataFrame (DatetimeIndex × symbols)}.
        # Only the fields in available_fields(); vwap/returns are synthesized
        # for you upstream. Align all frames on a shared index.
        ...
```

Register it so config can select it by name (`data/providers/__init__.py`
maintains a `PROVIDERS: dict[str, type[DataProvider]]` registry).

## Declaring capabilities (the tier system)

Factor gating depends entirely on `available_fields()`. The three tiers live in
`data/tiers.py`:

- `standard` — `open, high, low, close, volume` (`vwap`, `returns` synthesized).
- `fundamental` — adds `sector, industry, subindustry, cap`.
- `microstructure` — adds `effSpread, spread, orderFlow, hidden, depth, lobImb, …`.

Compose your `available_fields()` as the **union of the tiers you can fill**.
A daily yfinance/AlphaVantage vendor is `standard`. FMP can also reach
`fundamental`. LOBSTER reaches `microstructure`.

> Returning a field you can't reliably populate is worse than omitting it —
> factors that need it will be admitted, then produce garbage. Only advertise
> fields you actually deliver.

## Frequency

`fetch(..., freq)` receives a frequency string (`"1d"`, `"1min"`, `"10s"`, …).
Return data at that frequency. Downstream, `data/frequency.py` infers
annualization from the returned index — you don't compute it, but you **must**
return a clean monotonic `DatetimeIndex` so inference is correct.

## Caching

Don't implement caching inside the provider. `data/cache.py` wraps every
provider with a parquet cache keyed by `(provider, symbol, freq, asset_class)`.
Your `fetch()` should be a *pure* pull from the vendor; the cache layer handles
incremental ranges and rate limits.

## Checklist for a new provider

1. `data/providers/myvendor.py` implementing the ABC.
2. Add to the `PROVIDERS` registry.
3. Declare `available_fields()` via the tier helpers.
4. If it needs a key: read it from `.env` (e.g. `MYVENDOR_API_KEY`) and add it
   to the wizard's key-detection list (see [`ONBOARDING.md`](ONBOARDING.md)).
5. Add a small smoke test under `tests/` that `fetch()` returns the expected
   panel shape (mock the HTTP layer or guard on a key being present).
6. Note any quirks (adjusted vs raw prices, rate limits) in this file.

## Provider notes

| Provider | Key needed | Tier | Notes |
|----------|------------|------|-------|
| `lobster` | no (local CSVs) | `microstructure` | The original loader; `DATA_DIR=ticker_data`. |
| `yfinance` | no | `standard` | ✅ live. Daily OHLCV, `auto_adjust=True` (adjusted close). Network only in `_fetch`; parquet-cached under `data/market/`. Static preset lists carry survivorship bias; adjusted prices aren't point-in-time. |
| `fmp` | `FMP_API_KEY` | `standard` | ✅ live. FMP **stable** API `historical-price-eod/dividend-adjusted` → split/div-**adjusted** OHLC. (Legacy `/api/v3` returns 403 for keys issued after Aug 2025.) Generous free tier. |
| `alphavantage` | `ALPHAVANTAGE_API_KEY` | `standard` | ✅ live. `TIME_SERIES_DAILY`. **Free tier = UNADJUSTED + `compact` (~100 recent bars) + ~25 req/day + 5/min**; throttled to ~13s/req, cache essential. Prefer FMP/yfinance for longer/adjusted history. |

All three API providers extend `ApiProvider` (`base.py`) — they implement only
`available_fields()` and `_fetch(symbols)`; the universe→cache→assemble flow is
shared. The shared `_http.request_json` handles throttling + retry/backoff and
detects vendor rate-limit payloads.
