"""Convert a local raw LOBSTER folder → ``ticker_data/{TICKER}/bin{YYYYMM}.csv``.

This is the Phase-A entrypoint and replaces the exploratory ``convert_lobster.ipynb``
stub.  It is fully offline: point it at a downloaded LOBSTER folder (message +
orderbook CSVs per day) and it writes the 10-second-bar format the fund consumes.

Levels are inferred from the data, so the same command handles the level-3 GLD
sample on disk and the level-5 pulls downloaded later.

Examples
--------
::

    # Convert the bundled GLD level-3 sample (ticker inferred from the folder name).
    ./venv/bin/python scripts/convert_lobster.py --raw-dir GLD_2026-06-01_2026-06-11_3

    # Explicit ticker + output root.
    ./venv/bin/python scripts/convert_lobster.py \
        --raw-dir /scratch/lobster/AAPL_2024-01 --ticker AAPL --out-dir ticker_data
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

# Make ``quant_fund_agent`` importable when run as ``python scripts/convert_lobster.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_fund_agent.data.lobster_ingest.converter import convert_folder  # noqa: E402


def _infer_ticker(raw_dir: Path) -> str | None:
    """LOBSTER folders/files start with the ticker, e.g. ``GLD_2026-06-01_…``."""
    m = re.match(r"([A-Z][A-Z0-9.\-]*)_\d{4}-\d{2}-\d{2}", raw_dir.name)
    if m:
        return m.group(1)
    for msg in raw_dir.glob("*_message_*.csv"):
        m = re.match(r"([A-Z][A-Z0-9.\-]*)_\d{4}-\d{2}-\d{2}", msg.name)
        if m:
            return m.group(1)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True, help="folder of LOBSTER message/orderbook CSVs")
    ap.add_argument("--ticker", default=None, help="ticker symbol (default: inferred from folder/file name)")
    ap.add_argument("--out-dir", default="ticker_data", help="output root (default: ticker_data)")
    ap.add_argument(
        "--no-merge",
        action="store_true",
        help="overwrite monthly files instead of merging/de-duping with existing rows",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_dir():
        ap.error(f"--raw-dir not found: {raw_dir}")
    ticker = args.ticker or _infer_ticker(raw_dir)
    if not ticker:
        ap.error("could not infer --ticker from the folder; pass it explicitly")

    summary = convert_folder(
        raw_dir,
        ticker=ticker,
        out_dir=args.out_dir,
        merge_existing=not args.no_merge,
    )
    print(
        f"\n{ticker}: converted {summary.days} day(s) → {summary.rows} bar rows "
        f"across month(s) {', '.join(summary.months) or '—'}"
    )
    for p in summary.out_files:
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
