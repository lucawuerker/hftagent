#!/usr/bin/env python
"""Build point-in-time index membership tables from FMP's premium change logs.

Writes the same canonical interval table every consumer already reads::

    quant_fund_agent/data/universes/membership/{sp500,nasdaq100}.csv
        ticker,name,start_date,end_date,add_reason,remove_reason,source,cik

so ``data/membership.py``, ``resolve_universe`` and the per-bar panel mask work
unchanged — only the *source* of the table changes (FMP native instead of the
free public reconstruction).

Reconstruction is a **backward walk** of ``historical-{index}-constituent`` from
the current constituent list, run-length-encoded into spells; see
``quant_fund_agent/data/fmp_ingest/constituents.py``.  Because a backward walk
propagates every missing event into all earlier dates, the build is not trusted
on its own:

  * **audit** — month-end count band, no overlapping spells, ``start < end``;
  * **reconcile** — month-by-month Jaccard against the independent free
    reconstruction (``sp500_public.csv``), reported *per year* so a thin
    pre-2010 change log shows up instead of being averaged away.

Run::

    # 0. fetch the raw index payloads into the archive (once)
    ./venv/bin/python scripts/fmp_bulk_download.py --groups index

    # 1. build both tables, back to 2004
    ./venv/bin/python scripts/build_fmp_membership.py --index sp500,nasdaq100 --since 2004-01-01

    # build + report but write no canonical CSV
    ./venv/bin/python scripts/build_fmp_membership.py --dry-run

The previous curated S&P 500 table is preserved as ``sp500_public.csv`` (never
overwritten), so both reconstructions stay available and the diff is
reproducible.  See ``docs/data-layer/FMP_PREMIUM_ARCHIVE.md``.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_fund_agent.data.fmp_ingest.constituents import (  # noqa: E402
    INDEX_ENDPOINTS,
    build_intervals,
    to_csv_frame,
)
from quant_fund_agent.data.fmp_ingest.store import DEFAULT_ROOT, Archive  # noqa: E402
from quant_fund_agent.data.fmp_ingest.symbols import SYMBOL_MAP_FILE  # noqa: E402
from quant_fund_agent.data.membership import (  # noqa: E402
    MEMBERSHIP_DIR,
    audit_spells,
    compare_spells,
    load_membership,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("build_fmp_membership")

DEFAULT_SINCE = "2004-01-01"
#: the free reconstruction, preserved under this name as the cross-check
PUBLIC_TABLE = "sp500_public"


def _preserve_public_table() -> Path | None:
    """Copy the existing curated ``sp500.csv`` aside before FMP overwrites it."""
    src = MEMBERSHIP_DIR / "sp500.csv"
    dst = MEMBERSHIP_DIR / f"{PUBLIC_TABLE}.csv"
    if not src.exists() or dst.exists():
        return dst if dst.exists() else None
    shutil.copy2(src, dst)
    log.info("preserved curated table → %s", dst.relative_to(ROOT))
    return dst


def _spells_from_csv(index: str) -> pd.DataFrame | None:
    try:
        return load_membership(index)
    except FileNotFoundError:
        return None


def _yearly_jaccard(recon: pd.DataFrame) -> pd.DataFrame:
    """Collapse the month-end reconciliation to one row per year."""
    if recon.empty:
        return recon
    out = recon.copy()
    out["year"] = pd.to_datetime(out["date"]).dt.year
    return out.groupby("year").agg(
        months=("jaccard", "size"),
        mean_jaccard=("jaccard", "mean"),
        min_jaccard=("jaccard", "min"),
    ).round(4).reset_index()


def build_one(index: str, archive: Archive, since: str) -> dict:
    current_ep, changes_ep, band = INDEX_ENDPOINTS[index]
    current = archive.read(current_ep)
    changes = archive.read(changes_ep)
    if current is None or changes is None:
        raise FileNotFoundError(
            f"{index}: missing archived payloads ({current_ep} / {changes_ep}) under "
            f"{archive.root}. Run `scripts/fmp_bulk_download.py --groups index` first."
        )
    build = build_intervals(
        index,
        current.reset_index().to_dict("records"),
        changes.reset_index().to_dict("records"),
        since=since,
    )
    spells = build.spells
    errors, counts = audit_spells(spells, since, band=band)
    return {
        "index": index, "build": build, "spells": spells,
        "errors": errors, "counts": counts, "band": band,
    }


def _coverage_appendix(results: list[dict], archive: Archive) -> list[str]:
    """Which membership tickers the vendor could not price, and when they left.

    Read from the download's ``symbol_map.csv`` when it exists (the membership
    build normally runs *before* the price pull, so on a first build there is
    nothing to report yet).  This is the concrete, per-name form of the coverage
    percentages in ``FMP_PREMIUM_ARCHIVE.md`` §6 — the names a study over this
    window is missing, listed so the gap can be cited rather than estimated.
    """
    path = archive.root / SYMBOL_MAP_FILE
    if not path.exists():
        return []
    try:
        sm = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001 — a missing appendix must not fail a build
        log.warning("could not read %s: %s", path, e)
        return []

    names: dict[str, str] = {}
    membership: dict[str, set[str]] = {}
    for res in results:
        tag = {"sp500": "S", "nasdaq100": "N"}.get(res["index"], res["index"][:1].upper())
        for ticker, name in zip(res["spells"]["ticker"], res["spells"]["name"]):
            if isinstance(name, str) and name.strip():
                names.setdefault(ticker, name.strip())
            membership.setdefault(ticker, set()).add(tag)
    try:
        pub = load_membership(PUBLIC_TABLE)
        for ticker, name in zip(pub["ticker"], pub.get("name", pd.Series(dtype=str))):
            if isinstance(name, str) and name.strip():
                names.setdefault(ticker, name.strip())
            membership.setdefault(ticker, set()).add("P")
    except FileNotFoundError:
        pass

    unresolved = sm[sm["resolved"].isna()].copy()
    partial = sm[(sm["resolved"].notna()) & (sm["spell_coverage"] < 0.5)].copy()
    if unresolved.empty and partial.empty:
        return ["## Vendor coverage", "", "Every membership ticker was priced.", ""]

    unresolved["exit"] = pd.to_datetime(unresolved["spell_end"], errors="coerce")
    lines = [
        "## Vendor coverage — tickers with no usable price history",
        "",
        f"Of **{len(sm)}** tickers in the downloaded universe, **{len(unresolved)}** "
        f"returned no bars under any candidate symbol and **{len(partial)}** more "
        "were priced for less than half of their membership window.",
        "",
        "These are a **vendor limit, not a resolution bug**: FMP carries no security "
        "for them at all (`search-symbol` and `search-name` both come back empty for "
        "Bear Stearns, AT&T Wireless, Countrywide, Cephalon, Andrew, BEA Systems). "
        "Where a modern ticker exists for a post-bankruptcy successor (`ABKFQ` → "
        "`AMBC`) it is a **different security** and must not be spliced onto the old "
        "series.",
        "",
        "Index column: `S` = FMP S&P 500, `N` = Nasdaq-100, `P` = free "
        "reconstruction (`sp500_public`).",
        "",
        "### Unresolved, by the era they left the index",
        "",
    ]
    eras = ((2004, 2010, "left 2004–2009"), (2010, 2015, "left 2010–2014"),
            (2015, 2020, "left 2015–2019"), (2020, 2100, "left 2020 or later"))
    for lo, hi, label in eras:
        block = unresolved[(unresolved["exit"].dt.year >= lo)
                           & (unresolved["exit"].dt.year < hi)].sort_values("ticker")
        if block.empty:
            continue
        lines += [f"**{label} — {len(block)} names**", "",
                  "| ticker | company | membership window | index |",
                  "|---|---|---|---|"]
        for _, r in block.iterrows():
            company = names.get(r["ticker"], "—")
            idx = "".join(sorted(membership.get(r["ticker"], {"?"})))
            lines.append(
                f"| `{r['ticker']}` | {company} | {r['spell_start']} → {r['spell_end']} | {idx} |")
        lines.append("")
    undated = unresolved[unresolved["exit"].isna()]
    if len(undated):
        lines += [f"**No membership window recorded — {len(undated)} names:** "
                  + ", ".join(f"`{t}`" for t in sorted(undated["ticker"])), ""]

    if not partial.empty:
        partial = partial.sort_values("spell_coverage")
        # Zero coverage *with* bars means the vendor served a different company:
        # the ticker was reused after the original left the index (Pall Corp's
        # PLL is now Piedmont Lithium; Phelps Dodge's PD is now PagerDuty).
        reused = partial[(partial["spell_coverage"] <= 0.0) & (partial["n_bars"] > 0)]
        genuine = partial.drop(reused.index)

        def _table(block, heading, blurb):
            rows = [heading, "", blurb, "",
                    "| ticker | company | coverage | bars | vendor history "
                    "| membership window |", "|---|---|---|---|---|---|"]
            for _, r in block.iterrows():
                first, last = r.get("first_date"), r.get("last_date")
                vendor = ("—" if pd.isna(first) or pd.isna(last) else f"{first} → {last}")
                rows.append(
                    f"| `{r['ticker']}` | {names.get(r['ticker'], '—')} | "
                    f"{100 * float(r['spell_coverage']):.0f}% | {int(r['n_bars'])} | "
                    f"{vendor} | {r['spell_start']} → {r['spell_end']} |")
            return rows + [""]

        if not reused.empty:
            lines += _table(
                reused,
                "### Ticker reused by a different company",
                f"**{len(reused)} tickers** returned bars that do not overlap the "
                "membership window at all — the symbol was recycled after the "
                "original constituent left. **The point-in-time mask already "
                "excludes every one of these bars** (verified: zero survive "
                "`membership_mask`), so the panel is unaffected; they are listed "
                "because reading the archive *without* the mask would silently "
                "splice one company's prices onto another's history.")
        if not genuine.empty:
            lines += _table(
                genuine,
                "### Priced, but for under half of their membership window",
                "The vendor's history starts after the name joined the index. "
                "These bars are real and correctly masked; the earlier part of "
                "the spell is simply absent.")
    return lines


def write_report(path: Path, results: list[dict], since: str, recon: pd.DataFrame,
                 archive: Archive | None = None) -> None:
    lines = [
        "# FMP point-in-time membership build",
        "",
        f"- built: {date.today().isoformat()}",
        f"- since: {since}",
        f"- source: FMP `historical-*-constituent` (backward walk) + current constituent list",
        "",
    ]
    for res in results:
        build = res["build"]
        spells, counts = res["spells"], res["counts"]
        lines += [
            f"## {res['index']}",
            "",
            f"- spells: **{len(spells)}** over **{spells['ticker'].nunique()}** distinct tickers",
            f"- change log earliest event: **{build.log_earliest.date() if build.log_earliest is not None else 'n/a'}**",
            f"- still-active spells: {int(spells['end_date'].isna().sum())}",
            f"- left-censored names (spell start is a floor, not a fact): "
            f"{len(build.left_censored)}",
        ]
        if len(counts):
            lines.append(
                f"- month-end constituent count: min {counts['n'].min()}, "
                f"max {counts['n'].max()}, mean {counts['n'].mean():.1f} "
                f"(band {res['band']})"
            )
        if build.notes:
            lines += ["", "**Notes:**"] + [f"- {n}" for n in build.notes]
        if res["errors"]:
            lines += ["", "**AUDIT FAILURES:**"] + [f"- {e}" for e in res["errors"]]
        else:
            lines += ["", "Audit: all invariants passed."]
        lines.append("")

    if not recon.empty:
        lines += [
            "## Reconciliation vs the free public reconstruction (sp500_public)",
            "",
            "Per-year month-end Jaccard. A thin change log shows as a low early-year "
            "score; the free reconstruction remains the fallback for those years.",
            "",
            "| year | months | mean Jaccard | min Jaccard |",
            "|---|---|---|---|",
        ]
        for _, r in _yearly_jaccard(recon).iterrows():
            lines.append(
                f"| {int(r['year'])} | {int(r['months'])} | {r['mean_jaccard']:.3f} "
                f"| {r['min_jaccard']:.3f} |"
            )
        lines += [
            "",
            f"Overall mean Jaccard: **{recon['jaccard'].mean():.4f}** over "
            f"{len(recon)} month-ends.",
            "",
        ]
    if archive is not None:
        lines += _coverage_appendix(results, archive)
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default="sp500,nasdaq100",
                    help="comma list of indices to build (default sp500,nasdaq100)")
    ap.add_argument("--since", default=DEFAULT_SINCE,
                    help=f"drop spells ending before this date (default {DEFAULT_SINCE})")
    ap.add_argument("--archive", default=str(DEFAULT_ROOT), help="FMP archive root")
    ap.add_argument("--dry-run", action="store_true",
                    help="build + report, write no canonical CSV")
    args = ap.parse_args(argv)

    indices = [i.strip() for i in args.index.split(",") if i.strip()]
    unknown = [i for i in indices if i not in INDEX_ENDPOINTS]
    if unknown:
        ap.error(f"unknown index/indices {unknown}; known: {sorted(INDEX_ENDPOINTS)}")

    archive = Archive(args.archive)
    public = _preserve_public_table()

    results = []
    for index in indices:
        log.info("building %s …", index)
        res = build_one(index, archive, args.since)
        spells = res["spells"]
        log.info("  %d spells / %d tickers; log starts %s; %d left-censored",
                 len(spells), spells["ticker"].nunique(),
                 res["build"].log_earliest.date() if res["build"].log_earliest is not None else "?",
                 len(res["build"].left_censored))
        for e in res["errors"]:
            log.warning("  AUDIT: %s", e)
        for n in res["build"].notes:
            log.warning("  NOTE: %s", n)
        results.append(res)

    # Reconcile the S&P 500 build against the preserved free reconstruction.
    recon = pd.DataFrame()
    sp = next((r for r in results if r["index"] == "sp500"), None)
    if sp is not None and public is not None:
        reference = _spells_from_csv(PUBLIC_TABLE)
        if reference is not None:
            recon = compare_spells(sp["spells"], reference, args.since,
                                   label_a="fmp", label_b="public")
            if len(recon):
                log.info("reconcile vs public: mean Jaccard %.4f over %d month-ends",
                         recon["jaccard"].mean(), len(recon))
                log.info("\n%s", _yearly_jaccard(recon).to_string(index=False))

    report_path = MEMBERSHIP_DIR / "fmp_build_report.md"
    write_report(report_path, results, args.since, recon, archive=archive)
    log.info("report → %s", report_path.relative_to(ROOT))

    failed = any(r["errors"] for r in results)
    if args.dry_run:
        log.info("dry-run: not writing canonical CSVs.")
        return 1 if failed else 0

    for res in results:
        out = MEMBERSHIP_DIR / f"{res['index']}.csv"
        to_csv_frame(res["spells"]).to_csv(out, index=False)
        log.info("wrote %s (%d rows)", out.relative_to(ROOT), len(res["spells"]))
    load_membership.cache_clear()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
