#!/usr/bin/env python
"""One-time bulk download of the FMP premium archive for a PIT index universe.

Downloads full OHLCV (adjusted **and** unadjusted) plus the complete fundamental
record for **every name that was ever an index constituent** in the window, and
stores it verbatim as parquet under ``data/vendor/fmp/``.  Resumable, rate-limited
and safe to re-run: a completed unit of work is never re-fetched.

Typical sequence::

    # 0. what does this key actually serve?  (~30 calls)
    ./venv/bin/python scripts/fmp_bulk_download.py --probe

    # 1. the index change logs, so membership can be built
    ./venv/bin/python scripts/fmp_bulk_download.py --groups index
    ./venv/bin/python scripts/build_fmp_membership.py --since 2004-01-01

    # 2. cost estimate before spending bandwidth
    ./venv/bin/python scripts/fmp_bulk_download.py --dry-run

    # 3. the real pull (hours; resume by re-running the same command)
    ./venv/bin/python scripts/fmp_bulk_download.py --start 2004-01-01 --rate 600

Useful flags::

    --smoke 5            only the first N tickers (end-to-end check)
    --symbols AAPL,ATVI  explicit tickers instead of the PIT union
    --groups prices      restrict to one endpoint family
    --indices sp500      restrict the universe to one index
    --no-retry-errors    do not re-attempt units that errored on a previous run

Needs ``FMP_API_KEY`` in ``.env`` (Premium or above — the probe reports what is
gated).  See ``docs/data-layer/FMP_PREMIUM_ARCHIVE.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_fund_agent.data.fmp_ingest.capabilities import (  # noqa: E402
    CAPABILITIES_FILE,
    Capabilities,
    format_report,
    probe,
)
from quant_fund_agent.data.fmp_ingest.client import FMPClient  # noqa: E402
from quant_fund_agent.data.fmp_ingest.download import (  # noqa: E402
    DownloadConfig,
    Downloader,
)
from quant_fund_agent.data.fmp_ingest.endpoints import known_groups  # noqa: E402
from quant_fund_agent.data.fmp_ingest.store import DEFAULT_ROOT, Archive  # noqa: E402
from quant_fund_agent.data.fmp_ingest.symbols import (  # noqa: E402
    SYMBOL_MAP_FILE,
    load_symbol_changes,
    summarise,
    to_frame,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("fmp_bulk_download")

DEFAULT_START = "2004-01-01"
#: Stay under the Premium 750/min cap with headroom for retries.
DEFAULT_RATE = 600


def _load_env() -> None:
    """Read ``.env`` without a hard dependency on python-dotenv."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        return
    except ImportError:
        pass
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _spells(indices: list[str], start: str, end: str):
    """Concatenated membership spells for the configured indices (or None)."""
    import pandas as pd

    from quant_fund_agent.data.membership import load_membership

    frames = []
    for index in indices:
        try:
            frames.append(load_membership(index))
        except FileNotFoundError:
            log.warning("no membership table for %r yet — coverage will not be scored", index)
    return pd.concat(frames, ignore_index=True) if frames else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=DEFAULT_START, help=f"window start (default {DEFAULT_START})")
    ap.add_argument("--end", default=date.today().isoformat(), help="window end (default today)")
    ap.add_argument("--indices", default="sp500,nasdaq100",
                    help="PIT indices whose union forms the universe")
    ap.add_argument("--groups", default="prices,fundamentals,reference",
                    help=f"endpoint families to pull; known: {known_groups()}")
    ap.add_argument("--symbols", default="", help="explicit tickers (overrides --indices)")
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="only the first N tickers (end-to-end check)")
    ap.add_argument("--rate", type=int, default=DEFAULT_RATE, help="API calls per minute")
    ap.add_argument("--workers", type=int, default=8, help="concurrent symbols")
    ap.add_argument("--archive", default=str(DEFAULT_ROOT), help="archive root")
    ap.add_argument("--window-years", type=int, default=None,
                    help="force chunking of windowed endpoints into N-year requests")
    ap.add_argument("--probe", action="store_true",
                    help="probe plan capabilities, write capabilities.json, exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the cost estimate and exit without downloading")
    ap.add_argument("--no-retry-errors", action="store_true",
                    help="skip units that errored on a previous run instead of retrying")
    args = ap.parse_args(argv)

    _load_env()
    key = os.getenv("FMP_API_KEY", "")
    if not key:
        log.error("FMP_API_KEY not set in .env — add your premium key and re-run.")
        return 2

    archive = Archive(args.archive)
    client = FMPClient(key, per_minute=args.rate)
    groups = tuple(g.strip() for g in args.groups.split(",") if g.strip())
    indices = [i.strip() for i in args.indices.split(",") if i.strip()]

    # ── capability probe ────────────────────────────────────────────────────
    if args.probe:
        caps = probe(client, start=args.start, end=args.end)
        archive.write_json(CAPABILITIES_FILE, caps.to_dict())
        print(format_report(caps))
        print(f"\nwrote {archive.root / CAPABILITIES_FILE}  ({client.stats()['calls']} calls)")
        return 0

    stored = archive.read_json(CAPABILITIES_FILE)
    if stored is None:
        log.warning("no capabilities.json — run `--probe` first for an accurate plan; "
                    "assuming everything is served.")
        caps = Capabilities()
    else:
        caps = Capabilities.from_dict(stored)
        if caps.restricted():
            log.info("plan-restricted endpoints (skipped): %s", ", ".join(caps.restricted()))

    config = DownloadConfig(
        start=args.start, end=args.end, groups=groups, indices=tuple(indices),
        symbols=tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip()),
        max_symbols=args.smoke or None, workers=args.workers,
        retry_errors=not args.no_retry_errors, window_years=args.window_years,
    )

    # `index` is a run-level family; it has no per-symbol universe.
    if groups == ("index",):
        downloader = Downloader(client, archive, config, capabilities=caps)
        downloader.fetch_global()
        archive.compact_manifest({"client_stats": client.stats()})
        log.info("index payloads fetched: %s", client.stats())
        return 0

    spells = _spells(indices, args.start, args.end)
    downloader = Downloader(client, archive, config, capabilities=caps, spells=spells)
    symbols = downloader.universe()
    if not symbols:
        log.error("empty universe — build the membership tables first "
                  "(`scripts/build_fmp_membership.py`) or pass --symbols.")
        return 2

    plan = downloader.plan(symbols)
    log.info("PLAN: %d symbols x %d endpoints = %d units, %d HTTP calls",
             plan["symbols"], plan["endpoints"], plan["units"], plan["calls"])
    log.info("      ~%.2f GB at %d calls/min → ~%.0f min (%.1f h)",
             plan["estimated_gb"], plan["rate_per_minute"],
             plan["estimated_minutes"], plan["estimated_minutes"] / 60)
    if args.dry_run:
        # A dry run must cost nothing: the rename map is only needed for a real
        # pull, so it is fetched below rather than before the estimate.
        print(json.dumps(plan, indent=2))
        return 0

    if caps.allows("symbol_change"):
        downloader.renames = load_symbol_changes(client)
    report = downloader.run(symbols)

    # ── coverage report ─────────────────────────────────────────────────────
    resolutions = report.resolutions
    frame = to_frame(resolutions)
    (archive.root / SYMBOL_MAP_FILE).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(archive.root / SYMBOL_MAP_FILE, index=False)
    stats = summarise(resolutions)

    log.info("DONE  units done=%d skipped=%d  %s",
             report.units_done, report.units_skipped, report.by_status)
    log.info("      %d calls, %.2f GB downloaded, %.0f MB on disk",
             report.client_stats.get("calls", 0),
             report.client_stats.get("bytes", 0) / 1e9, report.archive_mb)
    log.info("      symbols resolved %d/%d (%.1f%%); coverage>=90%% for %d (%.1f%%)",
             stats["resolved"], stats["tickers"], stats["resolved_pct"],
             stats["coverage_ge_90pct"], stats["coverage_ge_90pct_pct"])
    if stats["unresolved"]:
        log.info("      unresolved: %s", ", ".join(stats["unresolved"][:40]))

    archive.write_json("coverage.json", {
        "window": {"start": args.start, "end": args.end},
        "indices": indices,
        "download": {
            "units_done": report.units_done, "units_skipped": report.units_skipped,
            "by_status": report.by_status, "client": report.client_stats,
            "archive_mb": report.archive_mb,
        },
        "resolution": stats,
    })
    log.info("symbol map → %s", archive.root / SYMBOL_MAP_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
