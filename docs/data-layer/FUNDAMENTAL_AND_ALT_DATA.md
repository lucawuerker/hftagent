# Beyond OHLCV — Fundamental & Alternative Data Fields (design doc)

> **Status: PROPOSED (not built).** Forward-looking spec for the data-layer stage
> *after* the multi-asset milestone (Phases 0–6). Companion to
> [`ARCHITECTURE.md`](ARCHITECTURE.md), [`DATA_PROVIDERS.md`](DATA_PROVIDERS.md),
> [`ROADMAP.md`](ROADMAP.md). FMP "stable" endpoint paths below should be
> **re-confirmed against the live docs at build time** — the site blocks
> scraping, so they're from working knowledge of the v3→stable migration, not a
> fresh fetch. AlphaVantage `function=` names are from its public docs.

## Why this matters

Today every factor reads OHLCV (`open/high/low/close/volume`, + synthesized
`vwap`/`returns`). That caps the Factor Researcher at price/volume technicals.
The biggest capability jump available is to let it build features on **non-price
data**: fundamentals (valuation, growth, quality), analyst estimates, corporate
events, sentiment/news, and macro. This is what turns the system from a
technical-signal generator into something that can express genuine cross-sectional
equity *theses* ("cheap, profitable, improving"). The whole data layer was built
tier-first precisely so this slots in: a provider advertises richer
`available_fields()`, factors declare richer `inputs`, and gating does the rest.

## Vendor inventory (what's actually available, free tier)

### FMP (stable API, `https://financialmodelingprep.com/stable/...`)
Per-symbol unless noted; all take `apikey`. Generous free tier.

| Family | Endpoint (stable) | Key fields |
|--------|-------------------|------------|
| Company profile | `profile?symbol=` | `sector, industry, marketCap, beta, exchange, country, ipoDate, isEtf` |
| Income statement | `income-statement?symbol=&period=annual\|quarter` | `revenue, grossProfit, operatingIncome, netIncome, eps, ebitda` |
| Balance sheet | `balance-sheet-statement?symbol=` | `totalAssets, totalLiabilities, totalEquity, cashAndEquivalents, totalDebt` |
| Cash flow | `cash-flow-statement?symbol=` | `operatingCashFlow, freeCashFlow, capitalExpenditure` |
| Key metrics | `key-metrics?symbol=&period=` | `peRatio, pbRatio, roe, roic, debtToEquity, currentRatio, marketCap` |
| Ratios | `ratios?symbol=` | profitability / liquidity / leverage / efficiency ratios |
| Analyst estimates | `analyst-estimates?symbol=` | revenue/EPS estimates per period (consensus) |
| Grades / targets | `grades?symbol=`, `price-target?symbol=` | rating changes, target prices |
| Insider trading | `insider-trading?symbol=` | transaction type, shares, price, date |
| Institutional | `institutional-ownership/...?symbol=` | holders, shares, change |
| Market cap (hist) | `historical-market-capitalization?symbol=` | daily `marketCap` series |
| Earnings/events | `earnings?symbol=`, `dividends?symbol=`, `splits?symbol=` | event dates + values |
| Macro (symbol-less) | `treasury-rates`, `economic-indicators?name=` | yield curve, GDP/CPI/unemployment |
| Sentiment / political | `social-sentiment`, `senate-trading`, `house-disclosure` | sentiment scores, congressional trades |

### AlphaVantage (`function=...`, free tier ≈ 5/min, ~25/day)
| Family | `function=` | Notes |
|--------|-------------|-------|
| Profile | `COMPANY_OVERVIEW` | sector, industry, marketcap, PERatio, DividendYield, …(point-in-time? **no** — latest snapshot) |
| Statements | `INCOME_STATEMENT`, `BALANCE_SHEET`, `CASH_FLOW`, `EARNINGS` | annual + quarterly arrays, `fiscalDateEnding` + `reportedDate` (EARNINGS) |
| Corporate | `DIVIDENDS`, `SPLITS`, `ETF_PROFILE`, `LISTING_STATUS`, `EARNINGS_CALENDAR`, `IPO_CALENDAR` | events / reference |
| Macro | `REAL_GDP`, `CPI`, `INFLATION`, `TREASURY_YIELD`, `FEDERAL_FUNDS_RATE`, `UNEMPLOYMENT`, `NONFARM_PAYROLL`, `RETAIL_SALES` | symbol-less time series |
| Alpha Intelligence | `NEWS_SENTIMENT`, `TOP_GAINERS_LOSERS`, `INSIDER_TRANSACTIONS`, `INSTITUTIONAL_HOLDINGS` | `NEWS_SENTIMENT` is the headline alt-data feed (per-ticker sentiment + relevance) |

> The free AV daily/throughput limits make it a *reference/enrichment* source,
> not something to poll across a big universe each run. FMP is the workhorse here.

## The hard problems (this is where the value is)

These aren't OHLCV-with-more-columns. Four issues must be solved deliberately, or
the agents will silently train on look-ahead-biased garbage.

### 1. Point-in-time / look-ahead — the central risk
Fundamentals are reported with a **lag** and get **restated**. A row dated
`fiscalDateEnding = 2023-03-31` was not *knowable* until the filing date (often
6–10 weeks later). Aligning a fundamental to its fiscal date leaks the future.
**Mandate:** every fundamental value is stamped at its **availability date**
(filing/`reportedDate`, or fiscal-end + a conservative reporting lag when the
filing date is absent) and only becomes visible to factors on/after that date —
then forward-filled. This must be enforced *in the data layer*, not left to each
factor. (The existing look-ahead invariant — load full, slice by
`as_of`/`cutoff` downstream — extends to this, but the **availability stamping**
is new and is the part that's easy to get wrong.)

### 2. Alignment onto the daily panel
The panel contract is a daily (or intraday) `DatetimeIndex`. Fundamentals are
quarterly *step functions*; events (earnings, insider trades) are sparse points;
macro is monthly/quarterly and **symbol-agnostic**. Each must be resampled to the
panel index by **forward-fill from the availability date** (a fundamental holds
until the next report). Decide the staleness cap (drop a value older than N
quarters → NaN → factor gated out for that name/date).

### 3. The panel-shape mismatch
The dict contract is `field -> wide DataFrame (dates × tickers)`. That fits
*time-varying per-ticker* fields (`peRatio`, `marketCap`) cleanly. But:
- **Static-ish per-ticker labels** (`sector`, `industry`) are what the Phase-4
  note flagged: 4 seed alphas pass `data["sector"]` (a wide frame) to
  `factors/ops.py::indneutralize`, which wants a **Series** (ticker→label). The
  first concrete deliverable should **fix `indneutralize`** to accept the wide
  frame (or add a panel→cross-section helper) so the `fundamental` tier works.
- **Symbol-agnostic macro** (CPI, yield curve) has *no ticker axis* — it breaks
  the wide-by-ticker shape. Options: (a) broadcast a macro series across all
  ticker columns (simple, wasteful), or (b) introduce a separate `macro` panel
  the factor API can read alongside the per-ticker panel. **Open decision.**

### 4. Caching & rate limits
Fundamentals move quarterly but the per-symbol/per-statement request count is
large (income + balance + cash-flow + metrics × N tickers). The parquet cache
generalizes — extend the key to `(provider, asset_class, field_group, symbol)` —
but with a **long TTL** (quarterly), unlike the daily-price cache. AV's ~25/day
free cap makes a one-time backfill + cache essential.

## How it slots into the existing tiers

The tier system already anticipates this — `data/tiers.py` defines
`fundamental = {sector, industry, subindustry, cap}`. The extension:
- enrich `fundamental` (add `pe, pb, roe, debtToEquity, revenue, eps, fcf, …`);
- add **new tiers** `estimates`, `sentiment`, `events`, `macro`;
- each provider advertises the groups it can fill (`FMP` → most; `AV` →
  fundamentals + macro + news-sentiment; `yfinance` → a little via `.info`);
- factors declare the new fields in `inputs` → `required_tier` resolves →
  `mcp/catalog_service` gates exactly as today. **No gating changes needed.**
- the Factor Researcher prompt must learn the new field vocabulary (what each
  field means, its frequency, and the look-ahead rule) so it generates correct,
  non-leaking factor code.

## Proposed phasing (smallest first slice)

1. **Profile + fundamental tier (FMP).** `profile` + `key-metrics` →
   `sector/industry/marketCap/peRatio/pbRatio/roe/...`, availability-stamped &
   forward-filled. **Fix `indneutralize`** + ship 2–3 example fundamental factors
   (e.g. value = `1/pe` cross-sectionally ranked, quality = `roe`). Proves the
   point-in-time + alignment machinery end-to-end on one vendor.
2. **Estimates + events.** `analyst-estimates`, `earnings` → revisions / surprise
   factors (these are where fundamentals get *alpha*, not just *beta*).
3. **Sentiment.** `NEWS_SENTIMENT` (AV) / `social-sentiment` (FMP) → a
   `sentiment` tier; low-frequency, cache-heavy.
4. **Macro.** Resolve the symbol-agnostic shape question (#3b) first; then yield
   curve / CPI as regime features.

## Open questions (decide before building)
- **Availability date source of truth** when a vendor omits the filing date — fixed
  reporting-lag (e.g. +45 trading days) vs vendor `reportedDate` only?
- **Macro shape**: broadcast across tickers vs a separate `macro` panel + factor-API
  change?
- **Restatement policy**: keep first-reported (true PIT) or accept the simpler
  latest-value (mild leak)?
- **Cross-vendor field normalization**: FMP `peRatio` vs AV `PERatio` — one
  canonical field name map (like `data/symbols.py` did for tickers)?
- **Cost/quota budgeting**: a backfill-once + long-TTL cache strategy, and which
  vendor is authoritative per field group.
