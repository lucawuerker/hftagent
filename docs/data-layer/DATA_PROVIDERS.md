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

## Multi-asset: crypto & FX (Phase 6)

All three API vendors serve `equity`, `crypto` and `fx`
(`asset_classes = ("equity","crypto","fx")`); `lobster` is `equity`-only.
`data/panel.py::get_provider` rejects an unsupported provider/asset-class combo
with a clear error before any fetch.

**Canonical symbols.** Universe presets and the panel use one provider-agnostic
symbol per instrument — equities are plain tickers (`AAPL`), crypto/fx are
`BASE-QUOTE` pairs (`BTC-USD`, `EUR-USD`). Each provider translates canonical →
native inside `_fetch` (via `data/symbols.py`) and returns frames **keyed by the
canonical symbol**, so the parquet cache and every downstream agent stay
vendor-agnostic. Presets: `crypto_demo`, `fx_demo`.

| Vendor | crypto | fx | Native form / endpoint |
|--------|--------|----|------------------------|
| `yfinance` | ✅ | ✅ | `BTC-USD` (unchanged) / `EURUSD=X`; same `yf.download`. |
| `fmp` | ✅ | ✅ | `BTCUSD`/`EURUSD` via `historical-price-eod/full` (**raw** OHLCV — no corporate actions for these). |
| `alphavantage` | ✅ | ✅ | `DIGITAL_CURRENCY_DAILY` (`symbol`+`market`) / `FX_DAILY` (`from_symbol`+`to_symbol`); free-tier rate-limited. |

**Caveats.**
- **Annualization is data-driven**: `data/frequency.py` infers 365 days/year when
  the index has weekend bars (crypto), else 252 — no `asset_class` plumbing.
- **FX has no reliable volume** from these vendors (filled `NaN`/0); volume-based
  factors are no-ops on an FX run. Prefer one `asset_class` per run.
- **FX weekend bars are vendor-dependent**: FMP stamps weekend FX bars (→ 365),
  yfinance/AV are weekday-only (→ 252). Each is annualized by its own sampling.

## Non-OHLCV fundamentals / estimates / events (Stage 7, equity-only)

A provider may also supply non-OHLCV fields by overriding
`ApiProvider._fetch_fundamentals(symbols)` → per-symbol **availability-stamped**
record frames; the base `load()` caches (`cache.py::cached_records`, quarterly
TTL) and aligns them onto the price index (`data/fundamentals.py`), then merges
into the panel. Declare the canonical fields you fill in `available_fields()`
(only what you actually deliver). Equity-only; `QF_FUNDAMENTALS=0` opts out. See
[`FUNDAMENTAL_AND_ALT_DATA.md`](FUNDAMENTAL_AND_ALT_DATA.md).

| Vendor | Fields delivered | Endpoints | Notes |
|--------|------------------|-----------|-------|
| `fmp` | sector, industry (free); marketCap, peRatio, pbRatio, psRatio, roe, roic, debtToEquity, currentRatio, grossMargin, netMargin, revenue, eps, freeCashFlow, epsEstimate, revenueEstimate, epsSurprise (**paid**) | `profile`, `key-metrics`, `ratios`, `income-statement`, `earnings` | Free tier returns only `profile`; the statement/ratio endpoints **402 Payment Required**. Degrades per-endpoint. |
| `alphavantage` | sector, industry, revenue, netMargin, eps, epsEstimate, epsSurprise | `OVERVIEW`, `INCOME_STATEMENT`, `EARNINGS` | **Free tier delivers these.** Only static labels are taken from the undated `OVERVIEW` snapshot; eps/surprise use the real `reportedDate`. Heavily rate-limited (≈25/day). |

**Caveats.**
- **Look-ahead is enforced in the data layer**: each value is stamped at its
  availability date (filing/`reportedDate`, else `fiscalDateEnding +
  reporting_lag_days`, default 60) and forward-filled with a staleness cap, so a
  fundamental is `NaN` before it was filed. The `_truncate_as_of` slice then
  needs no special handling.
- **Advertised ≠ delivered**: gating admits a factor when the provider advertises
  its field, but a key may not actually deliver it (FMP free tier) — factors must
  read `data.get("field")` and degrade to `NaN`, never `KeyError`.
- **No restatement vintages** on free tiers: the latest value per fiscal period is
  used (mild restatement leak); availability stamping is still conservative.
- AV's `OVERVIEW` ratios are a *current* snapshot (undated) → **not** backfilled
  (that would leak); only sector/industry are taken from it.
