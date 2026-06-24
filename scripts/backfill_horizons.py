#!/usr/bin/env python
"""Backfill ``prediction_horizon`` onto existing factor DB records.

Every ``FactorRecord`` now carries a ``prediction_horizon`` (in bars).  Legacy
records load with the schema default (6) but aren't *marked* as intentionally
set; this one-time migration stamps the horizon explicitly (idempotently, via a
``metadata["horizon_backfilled"]`` sentinel) across the main factor DB and every
prerun / workspace DB.

The user-chosen default is the **constant** strategy (everything → 6), which is
byte-identical to today's behaviour.  ``--strategy data_driven`` instead picks
each factor's best-|IC-IR| horizon from its already-measured ``ic_by_horizon``
grid (an in-sample pick — see the method docstring).

Usage::

    ./venv/bin/python scripts/backfill_horizons.py --dry-run
    ./venv/bin/python scripts/backfill_horizons.py            # writes
    ./venv/bin/python scripts/backfill_horizons.py --strategy data_driven --fill-suggested
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_fund_agent.databases import FactorDatabase  # noqa: E402


def _discover_dbs(root: Path) -> list[Path]:
    """Every ``factor_db.json`` under ``root`` (main + preruns + workspaces)."""
    return sorted(set(root.rglob("factor_db.json")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", choices=["constant", "data_driven"],
                    default="constant",
                    help="constant (default) → set every horizon to --value; "
                         "data_driven → best |IC-IR| from ic_by_horizon.")
    ap.add_argument("--value", type=int, default=6,
                    help="Horizon (bars) for the constant strategy / fallback.")
    ap.add_argument("--fill-suggested", action="store_true",
                    help="Also set suggested_horizons to the measured IC grid.")
    ap.add_argument("--force", action="store_true",
                    help="Re-stamp records already marked horizon_backfilled.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report per-factor horizons without writing.")
    ap.add_argument("--data-root", default="data",
                    help="Root to scan for factor_db.json files (default: data).")
    args = ap.parse_args()

    dbs = _discover_dbs(Path(args.data_root))
    if not dbs:
        print(f"No factor_db.json found under {args.data_root!r}.")
        return

    total = 0
    for path in dbs:
        db = FactorDatabase()
        db.load_from_json(path)
        if args.dry_run:
            # Recompute what would change without persisting the sentinel.
            n = db.backfill_prediction_horizons(
                strategy=args.strategy, value=args.value,
                fill_suggested=args.fill_suggested, force=True)
            horizons = sorted({r.prediction_horizon for r in db.list_factors()})
            print(f"[dry-run] {path}: {n} record(s), horizons → {horizons}")
        else:
            n = db.backfill_prediction_horizons(
                strategy=args.strategy, value=args.value,
                fill_suggested=args.fill_suggested, force=args.force)
            if n:
                db.save_to_json(path)
            print(f"{path}: updated {n} record(s)")
        total += n

    verb = "would update" if args.dry_run else "updated"
    print(f"\nDone: {verb} {total} record(s) across {len(dbs)} DB(s).")


if __name__ == "__main__":
    main()
