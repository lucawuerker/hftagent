# The FMP premium archive — a local, survivorship-bias-free dataset

This is the reproduction manual for the **one-time bulk download** that gives the
fund a local, point-in-time-correct S&P 500 + Nasdaq-100 dataset from **2004** to
today: full OHLCV (adjusted *and* unadjusted) plus the complete fundamental
record, for **every name that was ever a constituent** — including the ones that
were acquired, merged or went bankrupt.

It closes the two gaps the free path could not (see
[`SP500_MEMBERSHIP.md`](SP500_MEMBERSHIP.md) §7):

| gap on the free path | closed by |
|---|---|
| yfinance can't serve most **delisted** tickers (~82 % sampled coverage), so removed names silently vanish from the panel | FMP premium serves delisted symbols; the download measures per-ticker **spell coverage** and reports what is still missing |
| the live FMP provider normalises 5 endpoints down to **13 canonical fields** | the archive keeps **every** vendor field and exposes ~130 curated canonical ones |

---

## 1. Two layers

**Layer A — the raw archive** (`data/vendor/fmp/`, gitignored). Every column the
vendor returns, stored verbatim as parquet, one file per `(endpoint, symbol)`.
This is the source of truth: adding a canonical field later never means
re-downloading.

**Layer B — the panel provider**
(`quant_fund_agent/data/providers/fmp_archive.py`, `provider: fmp_archive`).
Reads the archive **offline** and emits the ordinary panel contract
`{field: DataFrame(index × tickers)}` with point-in-time availability stamping.
Because it sits behind the existing provider seam, every consumer — factor
research, the evolutionary loop, the Architect/Statistician, the walk-forward
trade loop, the comparison harness — inherits the richer, bias-free data with
**no per-consumer change**.

```
data/vendor/fmp/
  prices/adjusted/<SYM>.parquet        # historical-price-eod/dividend-adjusted
  prices/unadjusted/<SYM>.parquet      # historical-price-eod/full
  prices/dividends/<SYM>.parquet       prices/splits/<SYM>.parquet
  fundamentals/income-statement/{quarter,annual}/<SYM>.parquet
  fundamentals/{balance-sheet-statement,cash-flow-statement,ratios,
                key-metrics,financial-growth,enterprise-values}/…
  fundamentals/earnings/<SYM>.parquet
  reference/{profile,market-cap,shares-float}/<SYM>.parquet
  index/{sp500-constituent,historical-sp500-constituent,…}/_global.parquet
  manifest.json  manifest.jsonl  capabilities.json  symbol_map.csv  coverage.json
```

---

## 2. Running it

Needs `FMP_API_KEY` in `.env` on a **Premium** plan or above (750 calls/min,
30+ years of history, delisted symbols, quarterly periods; bulk endpoints are
Ultimate-only and are not used).

```bash
# 0. what does this key actually serve?  (~30 calls)
./venv/bin/python scripts/fmp_bulk_download.py --probe

# 1. index change logs → point-in-time membership tables
./venv/bin/python scripts/fmp_bulk_download.py --groups index
./venv/bin/python scripts/build_fmp_membership.py --index sp500,nasdaq100 --since 2004-01-01

# 2. cost estimate before spending bandwidth
./venv/bin/python scripts/fmp_bulk_download.py --dry-run

# 3. end-to-end check on a handful of names first
./venv/bin/python scripts/fmp_bulk_download.py --smoke 5

# 4. the real pull (hours) — re-run the same command to resume
./venv/bin/python scripts/fmp_bulk_download.py --start 2004-01-01 --rate 600
```

Then point a run at it: `--config quant.config.fmp_sp500.yaml` (or
`quant.config.nasdaq100.yaml`).

**Measured on the real run (2026-07-27, Premium, `--rate 600 --workers 8`):**
1 107 tickers (union of `sp500` ∪ `nasdaq100` ∪ `sp500_public`) → 24 354 units,
**~37 600 HTTP calls, ~3.5 GB downloaded, 1.0 GB of parquet on disk, ~55 min**
(20 symbols/min). 20 832 manifest rows: 19 375 `ok`, 1 456 `empty`, 1
`restricted`. FMP's Premium bandwidth allowance is 50 GB per trailing 30 days, so
a handful of full re-runs is affordable but not unlimited — which is why resume
matters. (That run was interrupted at 95 %; the resume re-fetched 687 units,
skipped 20 139, and finished in 90 seconds.)

### Resume

One **unit of work** is `(endpoint, period, symbol)` → one parquet file + one
manifest row. Outcomes are appended to `manifest.jsonl` (crash-safe under
concurrent workers) and compacted into `manifest.json` at the end. A killed run
re-enters exactly where it stopped; `ok` / `empty` / `restricted` units are never
re-fetched, `error` units are retried unless `--no-retry-errors`.

### Capability probe

FMP gates by plan at three levels, and the probe checks all three: whole
endpoints (402 *"Restricted Endpoint"*), parameter values (`period=quarter`, a
numeric cap on `limit`, `from`/`to` on `historical-market-capitalization`) and
**individual symbols** — delisted tickers answer 402 *"this value set for
'symbol' is not available under your current subscription"* on lower plans.
`capabilities.json` records the verdicts and the downloader spends calls only on
what is actually served. A restriction is terminal: it is never retried.

---

## 3. Membership: FMP native, publicly cross-checked

`scripts/build_fmp_membership.py` writes
`data/universes/membership/{sp500,nasdaq100}.csv` in the **existing canonical
schema** (`ticker,name,start_date,end_date,add_reason,remove_reason,source,cik`,
`end_date` exclusive), so `data/membership.py`, `resolve_universe` and the
per-bar mask work unchanged.

FMP serves membership as a *current list* plus a dated *change log*, so the table
is reconstructed by **backward-walking** the log from today's set:

```
S(n)   = today's constituents                      (effective from d_n)
S(k-1) = S(k) - added(d_k) + removed(d_k)          (effective on [d_{k-1}, d_k))
```

then run-length-encoding each ticker's presence into spells (a name that left and
rejoined gets two spells).

**Why it is not trusted on its own.** A backward walk propagates every *missing*
event into all earlier dates — the same weakness that made Wikipedia's change log
unusable pre-~2015 in the free build. So the build:

* **audits** — month-end count band (S&P 500 `[475, 515]`, Nasdaq-100
  `[95, 115]`), no overlapping spells per ticker, `start < end`;
* **reconciles** — month-by-month Jaccard against the free public
  reconstruction, reported **per year** in `fmp_build_report.md` so a thin
  early log shows up instead of being averaged away.

The free reconstruction is preserved as **`sp500_public.csv`** and rebuilt back
to 2004 (`build_sp500_membership.py --since 2004-01-01 --index sp500_public`),
which alone recovers **145 tickers** that were S&P 500 members between 2004 and
2010 and left before 2010 — Bear Stearns, Ambac, AT&T Wireless, Anheuser-Busch
and the rest of the 2008 casualty list. Switching primaries is a one-line config
change; both tables ship.

**Left censoring.** Names already in the index when the log begins have no add
event. They take `dateFirstAdded` from the current-constituent list when it is
available, otherwise a floor below both the log start and the requested window,
and are listed in the report as `left-censored` — their spell *start* is a floor,
not a fact. Their end dates, and every event inside the log, are unaffected.

### Symbol resolution

Membership tickers are the symbols a company carried **while it was a member**
(`AABA`, `ENRNQ`, `VIAC`, `FBHS`). Resolution is ordered and cheap: the literal
ticker first (it works for the vast majority), then `.`↔`-` variants, then the
`symbol-change` rename chain, then a stripped bankruptcy suffix
(`ENRNQ`→`ENRN`). Files are keyed by the **membership ticker**, not the resolved
vendor symbol, so the panel looks a name up by what it was called at the time.

`symbol_map.csv` records every attempt plus **spell coverage** — the fraction of
a ticker's actual membership window for which bars came back. That column is the
honest measure of how much survivorship bias was removed, and the number that
replaces §7's yfinance estimate.

---

## 4. Point-in-time discipline

Everything flows through the existing availability→forward-fill machinery in
`data/fundamentals.py`, so `modeling/service._truncate_as_of` keeps working
untouched. Three details are worth stating:

**Statements are stamped at their real filing date.** `income-statement`,
`balance-sheet-statement` and `cash-flow-statement` all carry `filingDate` and
`acceptedDate`.

**Unfiled endpoints inherit that date.** `ratios`, `key-metrics` and
`financial-growth` carry only the fiscal period end. Stamping them at
`period_end + 60 days` (the generic fallback) is both imprecise and occasionally
*leaking*. The archive provider joins them to the matching income-statement row
on `(fiscalYear, period)` and takes its **actual filing date**, falling back to
the lag only when no statement row matches. Where several filings match a
period, the **earliest** wins — a later restatement must not push the
availability date forward and hide a value that was genuinely public.

**Two fields are deliberately *not* PIT-backfilled.** `profile` (sector /
industry) is a *current* snapshot, so only its near-static labels are used, and
`shares-float` is likewise current-only — it is archived for reference but never
exposed as a panel field. The PIT-safe share count is
`weightedAverageShsOut` on the filing-stamped income statement, which *is*
exposed (`sharesOutstanding`).

**Adjusted vs unadjusted prices.** The panel serves the dividend-adjusted series
(correct for returns, and unchanged from the previous behaviour). The unadjusted
series, dividends and splits are archived alongside it so a **point-in-time**
adjustment factor can be rebuilt later — FMP's adjusted series is back-adjusted
using future corporate actions, which is harmless for returns but subtly
forward-looking for price-*level* factors.

---

## 5. What the researcher can now use

The canonical vocabulary lives in one table — `ARCHIVE_FIELD_SPECS` in
`quant_fund_agent/data/fields.py` — which drives three things that used to be
maintained separately and could drift: the tier sets (capability gating), the
per-endpoint normalisation maps, and the DATA CONTEXT prose in
`agents/factor_research/prompts.py`. Adding a field is one row.

~130 fields across: classification labels, size, income statement, balance
sheet, cash flow, profitability & returns, valuation multiples & yields,
leverage & coverage, liquidity, working-capital efficiency, per-share
quantities, growth rates, analyst estimates and earnings events. Because the
data-scope gate reads `available_fields()`, the Factor Researcher may now invent
factors over all of them — and `available_fields()` reports only endpoints that
are **actually on disk**, so a partial download gates factors out cleanly
instead of handing them NaN.

> **Units fix.** The legacy `FMP_METRICS_MAP` mapped the canonical
> `freeCashFlow` onto FMP's `freeCashFlowPerShare`, i.e. a field named for an
> absolute figure held a per-share one. `freeCashFlow` is now absolute USD
> everywhere (from the cash-flow statement) and `freeCashFlowPerShare` is its own
> field. Any factor written against the old meaning changes value.

---

## 6. How much survivorship bias actually went away

The honest metric is not "how many tickers resolved" but **what share of each
date's actual constituents the archive can price**. Measured on the built
archive (939 tickers with price files):

| date | S&P 500 members | priced | | Nasdaq-100 | priced |
|---|---|---|---|---|---|
| 2004-06-30 | 496 | **83.7 %** | | 101 | 79.2 % |
| 2008-06-30 | 500 | **88.8 %** | | 101 | 90.1 % |
| 2012-06-30 | 500 | **93.4 %** | | 102 | 95.1 % |
| 2016-06-30 | 505 | **98.2 %** | | 107 | 98.1 % |
| 2020-06-30 | 505 | **100 %** | | 105 | 100 % |
| 2026-06-30 | 503 | **100 %** | | 102 | 100 % |

So the panel is **effectively survivorship-bias-free from ~2016** and 84–93 %
complete before that. The residual is a **vendor coverage limit, not a resolution
bug**: FMP simply does not carry the securities that died in the 2008–2010 wave.
`search-symbol`/`search-name` return nothing for Bear Stearns (`BSC`), AT&T
Wireless (`AWE`), Countrywide (`CFC`), Cephalon (`CEPH`), Andrew (`ANDW`) or BEA
Systems (`BEAS`); `ABKFQ` (Ambac) resolves only to `AMBC`, the *post-bankruptcy*
entity, which is a different security and must not be spliced onto the old one.

Broken down by when a name left the index (FMP `sp500` table, 955 tickers):

| left the index | resolved |
|---|---|
| 2004–2009 | 81/143 (57 %) |
| 2010–2014 | 47/77 (61 %) |
| 2015–2019 | 92/118 (78 %) |
| 2020+ / still a member | 114/114 (**100 %**) |

**Implication for the thesis:** a study window starting in 2016 carries no
measurable survivorship bias from this dataset. A window reaching back to 2004
still under-represents 2008-era failures by ~15 % of names, which should be
stated rather than assumed away — and it is still far better than the free path,
whose missing names *are* exactly the delisted ones.

**Choice of membership table barely matters for coverage.** The FMP-native table
resolves 837/955 (87.6 %) and the free reconstruction 845/979 (86.3 %); they
disagree mostly on *ticker vintage*, not on membership. The archive was therefore
downloaded over the **union** of both, so switching which table masks the panel is
a config change, not a re-download.

**The full per-name list lives in
`quant_fund_agent/data/universes/membership/fmp_build_report.md`**, regenerated by
`build_fmp_membership.py` whenever `symbol_map.csv` is present: every unresolved
ticker with its company name and membership window, grouped by the era it left the
index, plus the reused-ticker and partial-coverage tables. Cite that file rather
than re-deriving the gap.

---

## 7. Limitations

* **Restatements.** The statement endpoints serve the *latest* version of each
  period, not the as-first-reported vintage. Availability stamping prevents the
  dominant look-ahead (seeing a quarter before it was filed) but not the smaller
  one (seeing a restated number). FMP's `*-as-reported` endpoints would close
  this; they are registered but not in the default groups.
* **The backward walk** is only as complete as FMP's change log — see §3.
* **No stable security master.** Identity is the normalised ticker plus a CIK
  where the current-constituent list supplies one. Ticker reuse is **real and
  common** — 95 of the 1 107 downloaded tickers return bars belonging to a later,
  different company (`PLL` Pall Corp → Piedmont Lithium, `PD` Phelps Dodge →
  PagerDuty, `ONE` Bank One, `NSM` National Semiconductor). The per-bar
  membership mask excludes all of them (their history lies entirely outside the
  original's spell — verified: zero bars survive), so anything loading through
  `load_panel` is safe; reading the archive directly is not. All 95 are tabulated
  in `universes/membership/fmp_build_report.md`.
* **Bandwidth.** 50 GB per trailing 30 days on Premium; a full re-pull is ~5 GB.
* **Licensing.** FMP data is licensed to the subscriber. `data/vendor/` is
  gitignored and must not be redistributed.
* **Load time.** A 6-field, 22-year, 837-ticker panel takes **~160 s** to
  assemble (837 symbols x several parquet reads + per-symbol availability
  alignment). Acceptable for a run that loads once and caches in the MCP research
  service; if repeated loads become the bottleneck, cache the assembled record
  frames per (symbol, field-set) rather than re-reading the statements.

---

## 8. Tests

Everything is offline, driven by payload shapes captured from live probes:

```bash
./venv/bin/pytest tests/test_fmp_ingest.py tests/test_fmp_archive_provider.py \
                  tests/test_membership.py -q
```

`test_fmp_ingest.py` covers the rate limiter, plan-restriction vs rate-limit vs
transient-error classification, parquet merge/dedup, manifest resume after a
simulated kill, window chunking, symbol-resolution fallbacks, the capability
probe's `limit`-cap parsing, and constituent reconstruction (rejoins, left
censoring, prose dates, an unparseable log).
`test_fmp_archive_provider.py` covers the panel path, with the load-bearing
assertions being the PIT ones: a value is NaN before its filing date, and the
unfiled endpoints inherit the statement's filing date rather than the lag.
