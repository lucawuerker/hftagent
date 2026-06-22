# S&P 500 membership build report

- generated: 2026-06-22T15:25:14+00:00
- snapshot: `quant_fund_agent/data/universes/membership/sources/20260622`
- since: 2010-01-01
- spells: 850 ; distinct tickers: 834
- still-active: 503

## Audit
- ✅ all invariants passed
- month-end count: min 497, max 506, mean 502.1 (band [475,515])
- cross-check vs fja05680 interval file: on [2010-01-01, today]: mine 834 tickers, theirs 841, only-mine ['FBIN'], only-theirs 8 (renames account for only-mine)

### Survivorship spot-checks
- TSLA member 2015-01-01? **False** (expect False; added 2020-12-21)
- TSLA member 2022-01-01? **True** (expect True)
- members on 2010-06-01: 499  |  2026-01-02: 503

## Reconciliation vs Wikipedia (month-end Jaccard)
- mean agreement: 0.9107 over 197 months (rises 0.84→0.99 as Wikipedia's 'Selected changes' log gets complete for recent years; fja05680 is the primary, Wikipedia the cross-check)
- recent 36 months mean: 0.9808
- worst month: 2010-01-31 (0.828)

## Renames coalesced
- curated: {'FB': 'META', 'RTN': 'RTX', 'BBT': 'TFC', 'FISV': 'FI', 'WLTW': 'WTW', 'ANTM': 'ELV', 'FBHS': 'FBIN'}
- auto-detected (same-day, same-company): {'BHI': 'BKR', 'DWDP': 'DD'}

## yfinance coverage (residual survivorship bias)
- not run (pass --check-coverage N).  Delisted names yfinance cannot serve are silently absent — closed by a premium provider (CRSP/FMP).
