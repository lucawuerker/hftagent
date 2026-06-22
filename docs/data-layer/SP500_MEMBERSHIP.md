# Survivorship-bias-free S&P 500 membership — how it was built

This document is the **reproduction manual** for the point-in-time (PIT) S&P 500
constituent set. It explains why the dataset exists, the exact free public
sources, the reconstruction + reconciliation algorithm, the audit, the known
limitations, and how to rebuild it from scratch (online or offline).

---

## 1. The problem

A *static* ticker list (e.g. `universes/sp100.txt`) applied across a 2010→today
backtest is **survivorship-biased**: it is the set of names that *survived* to the
snapshot date, so it over-weights winners and silently drops every company that
was an index member but later merged, was acquired, or went bankrupt (Activision,
Dell, Time Warner, Twitter/X, Monsanto, …). Backtests on the survivors look far
better than reality.

The fix is a **point-in-time membership table**: for every ticker, the dated
spell(s) during which it was actually an index constituent. The current universe
(503 names) becomes one slice of a much larger set — **834 distinct tickers were
in the S&P 500 at some point since 2010** (that 331-name difference *is* the
survivorship bias).

How top funds do this: a vendor PIT constituent feed (CRSP/Compustat via WRDS,
Bloomberg `INDX MEMB`, Refinitiv Datastream) keyed on a stable security id
(PERMNO/FIGI) that survives ticker renames, plus delisted price history. We
reproduce the **membership** half from free public sources here, and leave the
delisted-**price** half as a documented premium seam (§8).

---

## 2. Sources (free, public)

Both are fetched, snapshotted verbatim under
`quant_fund_agent/data/universes/membership/sources/<YYYYMMDD>/`, and reconciled.

| Source | Role | What it provides |
|---|---|---|
| **GitHub `fja05680/sp500`** | **primary** | `S&P 500 Historical Components & Changes (Updated).csv` — a `date → full constituent set` series, one row per change date, **1996-01-02 → present** (2 700+ rows). Also `sp500_ticker_start_end.csv`, a ready interval table we cross-check against. |
| **Wikipedia "List of S&P 500 companies"** | **reconciliation / enrichment** | The *current constituents* table (~503 names) and the *"Selected changes to the list"* change-log (dated add/remove events with reasons). Supplies company names, change reasons, rename detection, and an independent membership reconstruction. |

URLs (also recorded in each snapshot's `MANIFEST.json`, with byte counts, SHA-256,
and the pinned `fja05680` commit):

- `https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv`
- `https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv`
- `https://en.wikipedia.org/wiki/List_of_S%26P_500_companies`

**Independence caveat:** `fja05680` is itself partly derived from Wikipedia, so the
two are *not* fully independent. The reconciliation therefore catches
*parsing/transcription* errors and *recency lag*, not a shared upstream mistake —
the independent ground truth comes from the **audit invariants** (§5), not the
cross-source agreement alone.

---

## 3. Reconstruction algorithm (primary = fja05680)

`scripts/build_sp500_membership.py`:

1. **Parse** the components CSV into `(date, frozenset(tickers))` rows, sorted by
   date. Each row's set is effective on `[date, next_date)`.
2. **Forward run-length scan** (`intervals_from_components`): for each ticker,
   find maximal runs of consecutive rows in which it appears. A run over rows
   `[a..b]` becomes one spell `start = date[a]`, `end = date[b+1]` — the first date
   it is *no longer* a member (so `end_date` is **exclusive**, matching the index
   effective-date convention); a run that reaches the final row has `end = NaT`
   ("still a member"). A ticker that left and rejoined yields multiple spells.
3. **Renames** (`apply_renames`): relabel old→current ticker, then coalesce the
   now-contiguous spells into one. Sources of the rename map:
   - a small **curated** list (`KNOWN_RENAMES`: FB→META, RTN→RTX, BBT→TFC,
     FISV→FI, WLTW→WTW, ANTM→ELV, FBHS→FBIN); and
   - **auto-detected** same-day, same-company add/remove pairs from the Wikipedia
     change-log (`detect_renames`, e.g. BHI→BKR, DWDP→DD).

   Coalescing means a renamed name is **one continuous spell under its current
   ticker** (META: 2013-12-23→present), so a provider that serves the current
   ticker's full back-history (yfinance `META`) is masked *on* for the whole
   period — not just post-rename.
4. **Clip**: keep spells overlapping `[--since, today]` (default `2010-01-01`);
   spells that began earlier retain their *true* start date (informative, and the
   2010+ slice is identical either way).
5. **Enrich** (`enrich`): attach company `name`, and a change `reason` per spell
   endpoint. Wikipedia couples one add with one remove per row and the row's
   reason describes the **corporate action** — usually the *removed* name's
   acquisition, but for spin-off/split-off rows the *added* name's creation. We
   attribute each reason to the side it actually describes (spin-off → the added
   ticker; everything else → the removed ticker) so a counterparty's action is
   never pinned on the wrong name. Reasons are context only; **dates are the
   load-bearing facts**.

**Ticker normalization** (`normalize_ticker`): upper-case and `.`→`-`
(`BRK.B`→`BRK-B`, `BF.B`→`BF-B`) to match the yfinance/repo convention. Dead-ticker
suffixes (e.g. `…Q` bankruptcies, `AABA`) are kept verbatim — they are real
historical securities (and the ones yfinance can't serve; see §7).

Output: `quant_fund_agent/data/universes/membership/sp500.csv`
(`ticker,name,start_date,end_date,add_reason,remove_reason,source`), one row per
spell.

---

## 4. Reconciliation (cross-check = Wikipedia)

`WikiSnapshots` reconstructs membership independently by **backward-walking** the
Wikipedia change-log from today's constituent set: for each change date (newest→
oldest), the set *before* the change is `set_after − added + removed`. This yields
a step function `members_on(date)`. `reconcile` then compares it to the primary on
every **month-end** via the Jaccard index `|A∩B| / |A∪B|`.

Observed (2026-06-22 build): **mean Jaccard 0.911 over 197 months**, rising
monotonically **0.84 (2010) → 0.99 (2025)** (recent-36-month mean **0.981**). The
backward drift in older years is expected — Wikipedia's *"Selected changes"* log is
explicitly incomplete pre-~2015 — which is exactly why `fja05680` is the **primary**
and Wikipedia only the recency-weighted cross-check.

A second cross-check compares our reconstructed intervals to `fja05680`'s own
`ticker_start_end.csv` on the `[2010, today]` window: the only difference is one
ticker we relabel via a rename (`FBHS`→`FBIN`); 8 of 9 renames match theirs exactly.

---

## 5. Audit invariants

Checked in the build (`audit`) and re-asserted in `tests/test_membership.py`:

- **Count band** — `|members_as_of(d)| ∈ [475, 515]` at every month-end (the S&P
  500 briefly runs 500–505 with share classes / pending spin-offs). Observed:
  **min 497, max 506, mean 502.1** — all in band.
- **No overlapping spells** per ticker; every spell has `start < end`.
- **Still-active ≈ 503** (matches today's index incl. dual-class names).
- **Survivorship spot-checks**: TSLA absent 2015 / present 2022 (added 2020-12-21);
  AAPL present throughout; ATVI present 2016 / absent 2024 (Microsoft acquisition,
  removed 2023-10-18). These prove removed names are retained while members and
  dropped afterward.

---

## 6. How it plugs into the fund

`quant_fund_agent/data/membership.py` is the source-agnostic query API:
`members_as_of`, `union_members`, `membership_mask`, `apply_membership_mask`.

Turn it on with **one config field** — `DataSettings.membership: "sp500"` (in
`quant.config.yaml` under `data:`, or `QF_MEMBERSHIP=sp500`). Then:

1. `resolve_universe` (data/universe.py) returns the **union of every name ever a
   member** in `[start, end]`, so the provider fetches price history for names that
   later left.
2. `data.load_panel` (data/panel.py) applies a **per-bar boolean mask** once at
   load (`apply_membership_mask`), `NaN`-ing every `(date, ticker)` cell where the
   ticker was *not* a constituent on that date.

Because the mask is applied at the single panel-load seam, **every** consumer is
survivorship-correct for free: cross-sectional rank-IC / z-score / `indneutralize`
skip the `NaN`s (a name simply isn't in the cross-section on dates it wasn't a
member), the walk-forward trade loop only holds current constituents, and the
model-comparison harness inherits it too. No per-agent or per-loop change needed.

---

## 7. Limitations (free / tickers-only path)

- **Delisted prices (the residual bias).** yfinance cannot serve most *delisted*
  tickers (VIAC, WFM, X, AVP, …); those names are in the membership table but have
  no price column, so they silently drop out of the panel. The build's
  `--check-coverage N` probe quantifies this (≈ **82 %** of a 50-name sample served,
  a lower bound — some misses are rate-limit timeouts on live names like PFE/HLT).
  This is the one bias the free path cannot fully remove; a premium provider with
  delisted history closes it (§8). *(User decision: yfinance ⇒ tickers only.)*
- **Wikipedia incompleteness** for pre-~2015 changes — mitigated by using
  `fja05680` as primary; surfaced by the Jaccard trend.
- **Renames** are curated + same-day-heuristic detected; an unhandled rename shows
  as a one-segment discontinuity. The build report lists every applied/detected
  rename so the curated map can be extended.
- **No stable security master** (PERMNO/FIGI) — identity is the (normalized)
  ticker, so a ticker *reused* by a different company over time is a theoretical
  edge case (none observed in the S&P 500 2010+ set).
- **`auto_adjust` look-ahead** in yfinance prices is a pre-existing, documented
  data-layer limitation, orthogonal to membership.

---

## 8. Premium extension (closing the delisted-price gap)

`membership.py` defines a `MembershipSource` seam. The free
`PublicReconstructionSource` (this document) can be swapped for a vendor source
that supplies **both** PIT membership and delisted prices:

- **`CrspSource`** (WRDS/CRSP) — PIT index constituents keyed on PERMNO, with full
  delisted security history. Academic gold standard.
- **`FmpSource`** — FMP's historical-constituents endpoint + delisted price
  coverage (paid tier).

Both are stubbed (`NotImplementedError`) and only the *membership* table format
changes; the query API, the per-bar mask, and every downstream consumer are
unchanged.

---

## 9. Rebuild / reproduce

```bash
# Full rebuild from the live sources (writes sp500.csv + dated snapshot + report):
./venv/bin/python scripts/build_sp500_membership.py

# Validate without writing the canonical CSV:
./venv/bin/python scripts/build_sp500_membership.py --dry-run

# Reproduce offline from a saved snapshot (no network — true replay):
./venv/bin/python scripts/build_sp500_membership.py \
    --offline quant_fund_agent/data/universes/membership/sources/<YYYYMMDD>

# Quantify the yfinance delisted-coverage gap (probe N random union tickers):
./venv/bin/python scripts/build_sp500_membership.py --check-coverage 60

# Audit the shipped artifact:
./venv/bin/python -m pytest tests/test_membership.py -q
```

Each run writes `sources/<YYYYMMDD>/{<raw files>, MANIFEST.json, build_report.md}`
and a latest-copy `membership/build_report.md`. Re-fetch is skipped when the dated
snapshot already exists (use `--force` to refresh).
```yaml
# quant.config.yaml — enable PIT mode
data:
  provider: yfinance
  asset_class: equity
  start: '2010-01-01'
  end:   '2026-06-22'
  membership: sp500     # <- survivorship-bias-free, point-in-time universe
```
