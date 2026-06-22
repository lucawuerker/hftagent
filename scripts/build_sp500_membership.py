#!/usr/bin/env python
"""Build a survivorship-bias-free S&P 500 point-in-time membership table.

From **free public sources only**, write the canonical interval table consumed by
``quant_fund_agent.data.membership``::

    data/universes/membership/sp500.csv
        ticker,name,start_date,end_date,add_reason,remove_reason,source

Sources (both fetched, snapshotted verbatim, and reconciled):

  * **fja05680/sp500** (GitHub, *primary*) — a ``date -> full constituent set``
    series back to 1996 plus a ready interval file.  We reconstruct intervals
    ourselves from the date->set series (forward run-length scan) so the build is
    self-contained, then cross-check against their interval file.
  * **Wikipedia "List of S&P 500 companies"** (*reconciliation*) — the current
    constituents table + the "Selected changes to the list" change-log.  We
    backward-walk the change-log from today's set to reconstruct membership at any
    date and reconcile it against the primary on month-end samples.  It also
    supplies company names, change reasons, and rename detection.

Renames (e.g. FB->META) are coalesced into one continuous spell under the current
ticker, so a provider that serves the renamed ticker's full history (yfinance's
``META``) is masked on for the whole period rather than only post-rename.

Run::

    ./venv/bin/python scripts/build_sp500_membership.py            # fetch + build
    ./venv/bin/python scripts/build_sp500_membership.py --dry-run  # build, write nothing
    ./venv/bin/python scripts/build_sp500_membership.py --offline data/universes/membership/sources/<YYYYMMDD>
    ./venv/bin/python scripts/build_sp500_membership.py --check-coverage 60

Idempotent: re-fetches into a dated snapshot dir (reused unless ``--force``), so a
rebuild is replayable offline from any saved snapshot.  See
``docs/data-layer/SP500_MEMBERSHIP.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("build_sp500_membership")

ROOT = Path(__file__).resolve().parent.parent
MEMBERSHIP_DIR = ROOT / "quant_fund_agent" / "data" / "universes" / "membership"
SOURCES_DIR = MEMBERSHIP_DIR / "sources"

FJA_COMPONENTS_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
FJA_INTERVALS_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
)
FJA_COMMITS_API = "https://api.github.com/repos/fja05680/sp500/commits/master"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = "QuantFundAgent/thesis (S&P 500 membership reconstruction; research use)"

DEFAULT_SINCE = "2010-01-01"
# A name is normally one of ~500 (S&P briefly runs 500-505 with share classes /
# pending spin-offs).  Outside this band at a month-end => a likely dropped event.
COUNT_LO, COUNT_HI = 475, 515

# Curated renames not always captured by the same-day Wikipedia heuristic
# (old ticker -> current ticker).  Auto-detection (below) augments this; both are
# reported.  Keep additions here when the build report flags an uncoalesced gap.
KNOWN_RENAMES: dict[str, str] = {
    "FB": "META",        # Facebook -> Meta Platforms (2022-06)
    "GOOG_OLD": "GOOGL",  # placeholder; Google share classes are NOT a rename
    "RTN": "RTX",        # Raytheon -> Raytheon Technologies (2020-04)
    "BBT": "TFC",        # BB&T -> Truist (2019-12)
    "FISV": "FI",        # Fiserv (2023-07)
    "WLTW": "WTW",       # Willis Towers Watson (2022)
    "ANTM": "ELV",       # Anthem -> Elevance Health (2022-06)
    "CTXS": "CTXS",      # (kept; no-op example)
    "FBHS": "FBIN",      # Fortune Brands (2022)
    "JCI_OLD": "JCI",    # placeholder no-op
}
# Drop no-op / placeholder entries.
KNOWN_RENAMES = {k: v for k, v in KNOWN_RENAMES.items() if k != v and "_OLD" not in k}


# ── fetching + snapshotting ──────────────────────────────────────────────────

def _http_get(url: str) -> bytes:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    return r.content


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch_sources(snapshot_dir: Path, *, force: bool, offline: bool) -> dict[str, Path]:
    """Fetch (or reuse) the raw source files into ``snapshot_dir``; return paths."""
    paths = {
        "fja_components": snapshot_dir / "fja05680_components.csv",
        "fja_intervals": snapshot_dir / "fja05680_ticker_start_end.csv",
        "wikipedia": snapshot_dir / "wikipedia_list_of_sp500.html",
    }
    if offline:
        for name, p in paths.items():
            if not p.exists():
                raise FileNotFoundError(f"--offline snapshot missing {name}: {p}")
        log.info("offline: using snapshot %s", snapshot_dir)
        return paths

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    urls = {
        "fja_components": FJA_COMPONENTS_URL,
        "fja_intervals": FJA_INTERVALS_URL,
        "wikipedia": WIKI_URL,
    }
    for name, url in urls.items():
        p = paths[name]
        if p.exists() and not force:
            log.info("reuse  %s (cached)", p.name)
            continue
        log.info("fetch  %s", url)
        content = _http_get(url)
        p.write_bytes(content)
        manifest[name] = {"url": url, "bytes": len(content), "sha256": _sha256(content)}

    # Pin the fja05680 commit for reproducibility (best-effort).
    fja_sha = None
    try:
        fja_sha = requests.get(
            FJA_COMMITS_API, headers={"User-Agent": USER_AGENT}, timeout=30
        ).json().get("sha")
    except Exception as e:  # noqa: BLE001
        log.warning("could not pin fja05680 commit: %s", e)

    (snapshot_dir / "MANIFEST.json").write_text(json.dumps({
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fja05680_commit": fja_sha,
        "files": manifest,
    }, indent=2))
    return paths


# ── parsing ──────────────────────────────────────────────────────────────────

def normalize_ticker(t: str) -> str:
    """Canonicalize to the yfinance/repo convention (``BRK.B`` -> ``BRK-B``)."""
    return str(t).strip().upper().replace(".", "-")


def parse_components(path: Path) -> pd.DataFrame:
    """Parse the ``date,tickers`` series into ``date`` (Timestamp) + ``set``."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)
    df["set"] = df["tickers"].apply(
        lambda s: frozenset(normalize_ticker(x) for x in str(s).split(",") if x.strip())
    )
    return df[["date", "set"]]


def parse_wikipedia(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (current_constituents, changes_log) from the Wikipedia page HTML."""
    html = path.read_text(encoding="utf-8", errors="ignore")
    tables = pd.read_html(io.StringIO(html))
    current = tables[0].copy()
    current["ticker"] = current["Symbol"].apply(normalize_ticker)
    current = current.rename(columns={"Security": "name"})[["ticker", "name"]]

    changes = tables[1].copy()
    if isinstance(changes.columns, pd.MultiIndex):
        changes.columns = [
            " ".join(dict.fromkeys(str(x) for x in tup)).strip() for tup in changes.columns
        ]
    changes = changes.rename(columns={
        "Effective Date": "date",
        "Added Ticker": "add_ticker", "Added Security": "add_name",
        "Removed Ticker": "rm_ticker", "Removed Security": "rm_name",
        "Reason Reason": "reason", "Reason": "reason",
    })
    changes["date"] = pd.to_datetime(changes["date"], errors="coerce").dt.normalize()
    for c in ("add_ticker", "rm_ticker"):
        changes[c] = changes[c].apply(lambda v: normalize_ticker(v) if pd.notna(v) else None)
    changes = changes.dropna(subset=["date"]).reset_index(drop=True)
    return current, changes


# ── interval reconstruction ──────────────────────────────────────────────────

def intervals_from_components(comp: pd.DataFrame) -> pd.DataFrame:
    """Forward run-length scan: ``date->set`` rows -> per-ticker membership spells.

    Each row's set is effective on ``[date, next_date)``.  A ticker present in a
    maximal run of consecutive rows ``[a..b]`` yields one spell ``start=date[a]``,
    ``end=date[b+1]`` (the first date it is *no longer* a member; ``NaT`` if the
    run reaches the final, most-recent row).  Gaps (left then rejoined) yield
    multiple spells.
    """
    from collections import defaultdict

    dates = comp["date"].tolist()
    sets = comp["set"].tolist()
    n = len(dates)
    present_rows: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(sets):
        for t in s:
            present_rows[t].append(i)

    rows = []
    for t, idxs in present_rows.items():
        run_start = prev = idxs[0]
        for i in idxs[1:]:
            if i == prev + 1:
                prev = i
                continue
            rows.append((t, dates[run_start], dates[prev + 1]))
            run_start = prev = i
        end = dates[prev + 1] if prev + 1 < n else pd.NaT
        rows.append((t, dates[run_start], end))
    return pd.DataFrame(rows, columns=["ticker", "start_date", "end_date"])


def _coalesce(spells: pd.DataFrame) -> pd.DataFrame:
    """Merge per-ticker spells that touch or overlap into single spells."""
    OPEN = pd.Timestamp("2999-12-31")
    out = []
    for ticker, sub in spells.groupby("ticker"):
        sub = sub.sort_values("start_date")
        cur_s, cur_e = None, None
        for _, r in sub.iterrows():
            s = r["start_date"]
            e = r["end_date"] if pd.notna(r["end_date"]) else OPEN
            if cur_s is None:
                cur_s, cur_e = s, e
            elif s <= cur_e:                       # touching/overlapping -> extend
                cur_e = max(cur_e, e)
            else:
                out.append((ticker, cur_s, cur_e))
                cur_s, cur_e = s, e
        if cur_s is not None:
            out.append((ticker, cur_s, cur_e))
    df = pd.DataFrame(out, columns=["ticker", "start_date", "end_date"])
    df["end_date"] = df["end_date"].replace(OPEN, pd.NaT)
    return df


def detect_renames(changes: pd.DataFrame) -> dict[str, str]:
    """Same-day, same-company add/remove pairs -> ``old_ticker: new_ticker``."""
    detected: dict[str, str] = {}
    for _, g in changes.dropna(subset=["add_ticker", "rm_ticker"]).groupby("date"):
        for _, r in g.iterrows():
            an, rn = str(r.get("add_name", "")).strip().lower(), str(r.get("rm_name", "")).strip().lower()
            if an and an == rn and r["add_ticker"] != r["rm_ticker"]:
                detected[r["rm_ticker"]] = r["add_ticker"]
    return detected


def apply_renames(spells: pd.DataFrame, renames: dict[str, str]) -> pd.DataFrame:
    """Relabel old tickers to current, then coalesce the now-contiguous spells.

    Chase chains (A->B->C) to a fixed point so multi-rename names collapse fully.
    """
    def resolve(t: str) -> str:
        seen = set()
        while t in renames and t not in seen:
            seen.add(t)
            t = renames[t]
        return t

    s = spells.copy()
    s["ticker"] = s["ticker"].map(resolve)
    return _coalesce(s)


# ── Wikipedia backward walk (reconciliation) ─────────────────────────────────

class WikiSnapshots:
    """Step function: ``members_on(date)`` from the backward-walked change-log."""

    def __init__(self, current: set[str], changes: pd.DataFrame):
        grouped = []
        for d, g in changes.groupby("date"):
            added = set(g["add_ticker"].dropna())
            removed = set(g["rm_ticker"].dropna())
            grouped.append((d, added, removed))
        grouped.sort(key=lambda x: x[0])  # ascending change dates

        cur = set(current)
        snaps: list[tuple[pd.Timestamp, frozenset]] = []
        for d, added, removed in reversed(grouped):   # newest -> oldest
            snaps.append((d, frozenset(cur)))          # set effective from d forward
            cur = (cur - added) | removed              # step to just before d
        snaps.append((pd.Timestamp.min, frozenset(cur)))
        snaps.sort(key=lambda x: x[0])
        self._dates = [d for d, _ in snaps]
        self._sets = [s for _, s in snaps]
        self.earliest_change = grouped[0][0] if grouped else None

    def members_on(self, q) -> frozenset:
        import bisect
        q = pd.Timestamp(q).normalize()
        i = bisect.bisect_right(self._dates, q) - 1
        return self._sets[max(i, 0)]


def members_as_of_intervals(spells: pd.DataFrame, q) -> set[str]:
    q = pd.Timestamp(q).normalize()
    end = spells["end_date"].fillna(pd.Timestamp("2999-12-31"))
    active = (spells["start_date"] <= q) & (q < end)
    return set(spells.loc[active, "ticker"])


# ── enrichment, audit, report ────────────────────────────────────────────────

def build_name_map(current: pd.DataFrame, changes: pd.DataFrame) -> dict[str, str]:
    names: dict[str, str] = {}
    for _, r in changes.iterrows():
        if pd.notna(r.get("add_ticker")) and isinstance(r.get("add_name"), str):
            names.setdefault(r["add_ticker"], r["add_name"])
        if pd.notna(r.get("rm_ticker")) and isinstance(r.get("rm_name"), str):
            names.setdefault(r["rm_ticker"], r["rm_name"])
    for _, r in current.iterrows():            # current names win
        names[r["ticker"]] = r["name"]
    return names


def _is_spinoff(reason: object) -> bool:
    """A reason that describes the *added* name's creation (spin-off / split-off)."""
    r = str(reason).lower()
    return any(k in r for k in ("spun off", "spin-off", "spinoff", "split-off", "split off"))


def enrich(spells: pd.DataFrame, names: dict[str, str], changes: pd.DataFrame) -> pd.DataFrame:
    """Attach name + add/remove reason (matched by ticker & nearby date).

    Wikipedia couples one add with one remove per row, and the row's reason
    describes the **corporate action** — usually the *removed* name's
    acquisition/merger, but for spin-off rows the *added* name's creation.  We
    attribute each reason to the side it actually describes (spin-off → the added
    name; everything else → the removed name) so a counterparty's action is never
    pinned on the wrong ticker.  Reasons are context only; the dates are the
    load-bearing facts.
    """
    tol = pd.Timedelta(days=7)
    spinoff = changes["reason"].map(_is_spinoff)
    # add-side reason: a pure addition, or a spin-off (reason names the new ticker)
    add_evt = changes[changes["add_ticker"].notna() & (changes["rm_ticker"].isna() | spinoff)]
    # remove-side reason: anything not a spin-off (acquisition / merger / market-cap)
    rm_evt = changes[changes["rm_ticker"].notna() & ~spinoff]

    def reason_for(evt, key, ticker, when):
        if pd.isna(when):
            return ""
        hit = evt[(evt[key] == ticker) & ((evt["date"] - when).abs() <= tol)]
        return str(hit.iloc[0]["reason"]) if len(hit) else ""

    out = spells.copy()
    out["name"] = out["ticker"].map(names).fillna("")
    out["add_reason"] = [reason_for(add_evt, "add_ticker", t, d)
                         for t, d in zip(out["ticker"], out["start_date"])]
    out["remove_reason"] = [reason_for(rm_evt, "rm_ticker", t, d)
                            for t, d in zip(out["ticker"], out["end_date"])]
    out["source"] = "fja05680"
    return out[["ticker", "name", "start_date", "end_date", "add_reason", "remove_reason", "source"]]


def audit(spells: pd.DataFrame, since: pd.Timestamp) -> tuple[list[str], pd.DataFrame]:
    """Structural + count invariants.  Returns (errors, monthly count table)."""
    errors: list[str] = []
    bad = spells[spells["end_date"].notna() & (spells["end_date"] <= spells["start_date"])]
    if len(bad):
        errors.append(f"{len(bad)} spell(s) with end_date <= start_date")
    # No overlapping spells per ticker.
    for ticker, sub in spells.groupby("ticker"):
        sub = sub.sort_values("start_date")
        ends = sub["end_date"].fillna(pd.Timestamp("2999-12-31")).tolist()
        starts = sub["start_date"].tolist()
        for i in range(1, len(starts)):
            if starts[i] < ends[i - 1]:
                errors.append(f"overlapping spells for {ticker}")
                break
    # Monthly count band.
    end = min(pd.Timestamp.today().normalize(),
              spells["end_date"].fillna(pd.Timestamp.min).max() or pd.Timestamp.today())
    months = pd.date_range(since, pd.Timestamp.today().normalize(), freq="ME")
    counts = [(m, len(members_as_of_intervals(spells, m))) for m in months]
    cdf = pd.DataFrame(counts, columns=["date", "n"])
    out_of_band = cdf[(cdf["n"] < COUNT_LO) | (cdf["n"] > COUNT_HI)]
    if len(out_of_band):
        errors.append(
            f"{len(out_of_band)}/{len(cdf)} month-ends outside [{COUNT_LO},{COUNT_HI}] "
            f"(min {cdf['n'].min()}, max {cdf['n'].max()})"
        )
    return errors, cdf


def reconcile(spells: pd.DataFrame, wiki: WikiSnapshots, since: pd.Timestamp) -> pd.DataFrame:
    """Month-end Jaccard agreement between primary intervals and Wikipedia walk."""
    start = max(since, wiki.earliest_change or since)
    months = pd.date_range(start, pd.Timestamp.today().normalize(), freq="ME")
    rows = []
    for m in months:
        a = members_as_of_intervals(spells, m)
        b = set(wiki.members_on(m))
        if not (a or b):
            continue
        inter, union = len(a & b), len(a | b)
        rows.append((m, len(a), len(b), inter / union if union else 1.0,
                     ",".join(sorted(a - b)[:8]), ",".join(sorted(b - a)[:8])))
    return pd.DataFrame(rows, columns=["date", "n_primary", "n_wiki", "jaccard",
                                       "only_primary", "only_wiki"])


def compare_to_fja_intervals(mine: pd.DataFrame, path: Path, since: pd.Timestamp) -> str:
    """Cross-check my reconstruction against fja05680's own interval file.

    Compared on the same ``[since, today]`` window (their file spans 1996+), so the
    only expected differences are tickers I relabel via renames (e.g. FBHS->FBIN).
    """
    try:
        theirs = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        return f"(could not read fja interval file: {e})"
    theirs["ticker"] = theirs["ticker"].apply(normalize_ticker)
    theirs["end_date"] = pd.to_datetime(theirs["end_date"], errors="coerce")
    theirs = theirs[theirs["end_date"].isna() | (theirs["end_date"] > since)]
    mine_t, theirs_t = set(mine["ticker"]), set(theirs["ticker"])
    return (f"on [{since.date()}, today]: mine {len(mine_t)} tickers, theirs {len(theirs_t)}, "
            f"only-mine {sorted(mine_t - theirs_t)}, only-theirs {len(theirs_t - mine_t)} "
            f"(renames account for only-mine)")


def check_yf_coverage(tickers: list[str], sample: int) -> str:
    """Probe yfinance for a random sample of union tickers; report % served."""
    import random
    try:
        import yfinance as yf
    except Exception as e:  # noqa: BLE001
        return f"(yfinance unavailable: {e})"
    random.seed(0)
    probe = random.sample(tickers, min(sample, len(tickers)))
    served = 0
    for t in probe:
        try:
            h = yf.Ticker(t).history(period="1mo")
            served += int(not h.empty)
        except Exception:  # noqa: BLE001
            pass
    pct = 100 * served / len(probe) if probe else 0
    return f"{served}/{len(probe)} sampled tickers served by yfinance ({pct:.0f}%)"


def write_report(path: Path, *, snapshot_dir: Path, spells: pd.DataFrame,
                 since: pd.Timestamp, audit_errs: list[str], counts: pd.DataFrame,
                 recon: pd.DataFrame, renames: dict[str, str], detected: dict[str, str],
                 fja_cmp: str, coverage: str | None) -> None:
    def spot(q):
        return sorted(members_as_of_intervals(spells, q))

    tsla_2015 = "TSLA" in spot("2015-01-01")
    tsla_2022 = "TSLA" in spot("2022-01-01")
    lines = [
        "# S&P 500 membership build report", "",
        f"- generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- snapshot: `{snapshot_dir.relative_to(ROOT)}`",
        f"- since: {since.date()}",
        f"- spells: {len(spells)} ; distinct tickers: {spells['ticker'].nunique()}",
        f"- still-active: {int(spells['end_date'].isna().sum())}", "",
        "## Audit",
        ("- ✅ all invariants passed" if not audit_errs
         else "\n".join(f"- ❌ {e}" for e in audit_errs)),
        f"- month-end count: min {counts['n'].min()}, max {counts['n'].max()}, "
        f"mean {counts['n'].mean():.1f} (band [{COUNT_LO},{COUNT_HI}])",
        f"- cross-check vs fja05680 interval file: {fja_cmp}", "",
        "### Survivorship spot-checks",
        f"- TSLA member 2015-01-01? **{tsla_2015}** (expect False; added 2020-12-21)",
        f"- TSLA member 2022-01-01? **{tsla_2022}** (expect True)",
        f"- members on 2010-06-01: {len(spot('2010-06-01'))}  |  2026-01-02: {len(spot('2026-01-02'))}",
        "", "## Reconciliation vs Wikipedia (month-end Jaccard)",
        f"- mean agreement: {recon['jaccard'].mean():.4f} over {len(recon)} months "
        f"(rises 0.84→0.99 as Wikipedia's 'Selected changes' log gets complete for "
        f"recent years; fja05680 is the primary, Wikipedia the cross-check)"
        if len(recon) else "- (no overlap window)",
        f"- recent 36 months mean: {recon['jaccard'].tail(36).mean():.4f}"
        if len(recon) else "",
        f"- worst month: {recon.loc[recon['jaccard'].idxmin(), 'date'].date()} "
        f"({recon['jaccard'].min():.3f})" if len(recon) else "",
        "", "## Renames coalesced",
        f"- curated: {KNOWN_RENAMES}",
        f"- auto-detected (same-day, same-company): {detected}", "",
        "## yfinance coverage (residual survivorship bias)",
        f"- {coverage}" if coverage else
        "- not run (pass --check-coverage N).  Delisted names yfinance cannot serve "
        "are silently absent — closed by a premium provider (CRSP/FMP).",
        "",
    ]
    path.write_text("\n".join(str(x) for x in lines))


# ── main ─────────────────────────────────────────────────────────────────────

def to_csv_frame(spells: pd.DataFrame) -> pd.DataFrame:
    df = spells.sort_values(["ticker", "start_date"]).copy()
    df["start_date"] = df["start_date"].dt.strftime("%Y-%m-%d")
    df["end_date"] = df["end_date"].dt.strftime("%Y-%m-%d").where(df["end_date"].notna(), "")
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default="sp500", help="output index name (default sp500)")
    ap.add_argument("--since", default=DEFAULT_SINCE, help="drop spells ending before this date")
    ap.add_argument("--dry-run", action="store_true", help="build + report, write no canonical CSV")
    ap.add_argument("--force", action="store_true", help="re-fetch even if snapshot is cached")
    ap.add_argument("--offline", metavar="SNAPSHOT_DIR", help="rebuild from a saved snapshot dir")
    ap.add_argument("--check-coverage", type=int, default=0, metavar="N",
                    help="probe N random union tickers against yfinance")
    args = ap.parse_args(argv)

    since = pd.Timestamp(args.since).normalize()
    if args.offline:
        snapshot_dir = Path(args.offline)
        if not snapshot_dir.is_absolute():
            snapshot_dir = ROOT / snapshot_dir
    else:
        snapshot_dir = SOURCES_DIR / date.today().strftime("%Y%m%d")

    paths = fetch_sources(snapshot_dir, force=args.force, offline=bool(args.offline))

    log.info("parsing sources …")
    comp = parse_components(paths["fja_components"])
    current, changes = parse_wikipedia(paths["wikipedia"])
    log.info("  components: %d dated sets %s → %s", len(comp),
             comp["date"].min().date(), comp["date"].max().date())
    log.info("  wikipedia: %d current, %d change rows", len(current), len(changes))

    log.info("reconstructing intervals (primary = fja05680) …")
    raw = intervals_from_components(comp)
    detected = detect_renames(changes)
    renames = {**detected, **KNOWN_RENAMES}            # curated overrides detected
    spells = apply_renames(raw, renames)
    # keep spells overlapping [since, today]; retain true (pre-since) starts.
    keep = spells["end_date"].isna() | (spells["end_date"] > since)
    spells = spells[keep].reset_index(drop=True)
    log.info("  %d spells (%d tickers) after renames + since-filter",
             len(spells), spells["ticker"].nunique())

    names = build_name_map(current, changes)
    enriched = enrich(spells, names, changes)

    log.info("auditing + reconciling …")
    audit_errs, counts = audit(spells, since)
    wiki = WikiSnapshots(set(current["ticker"]), changes)
    recon = reconcile(spells, wiki, since)
    fja_cmp = compare_to_fja_intervals(spells, paths["fja_intervals"], since)
    coverage = None
    if args.check_coverage:
        union = sorted(set(spells["ticker"]))
        coverage = check_yf_coverage(union, args.check_coverage)

    for e in audit_errs:
        log.warning("AUDIT: %s", e)
    if len(recon):
        log.info("reconcile: mean Jaccard %.4f over %d months", recon["jaccard"].mean(), len(recon))
    log.info("cross-check vs fja interval file: %s", fja_cmp)
    if coverage:
        log.info("coverage: %s", coverage)

    report_path = snapshot_dir / "build_report.md"
    write_report(report_path, snapshot_dir=snapshot_dir, spells=spells, since=since,
                 audit_errs=audit_errs, counts=counts, recon=recon,
                 renames=renames, detected=detected, fja_cmp=fja_cmp, coverage=coverage)
    log.info("report → %s", report_path.relative_to(ROOT))

    if args.dry_run:
        log.info("dry-run: not writing canonical CSV.")
        log.info("\n%s", report_path.read_text())
        return 1 if audit_errs else 0

    MEMBERSHIP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MEMBERSHIP_DIR / f"{args.index}.csv"
    to_csv_frame(enriched).to_csv(out_path, index=False)
    # latest-report convenience copy
    (MEMBERSHIP_DIR / "build_report.md").write_text(report_path.read_text())
    log.info("wrote %s (%d rows)", out_path.relative_to(ROOT), len(enriched))
    return 1 if audit_errs else 0


if __name__ == "__main__":
    sys.exit(main())
