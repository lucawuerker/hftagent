"""Backfill capability-tier metadata onto an existing factor DB.

Fills ``required_inputs`` / ``required_tier`` on every record (from the factor
registry) so the Selector catalog can gate by provider without re-instantiating
factors, and so fresh clones — where researcher ``.py`` files are gitignored —
still gate correctly off the persisted metadata.

Usage:
    ./venv/bin/python scripts/annotate_factor_tiers.py [path/to/factor_db.json]

Idempotent: only records missing the metadata are touched (pass --force to redo).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from quant_fund_agent.databases.factor_db import FactorDatabase

DEFAULT_PATH = Path("data/factors/factor_db.json")


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--force"]
    force = "--force" in sys.argv[1:]
    path = Path(args[0]) if args else DEFAULT_PATH

    if not path.exists():
        raise SystemExit(f"factor DB not found: {path}")

    db = FactorDatabase()
    db.load_from_json(path)
    before = len(db.list_factors())
    updated = db.annotate_tiers(force=force)
    db.save_to_json(path)

    tiers = Counter(f.required_tier for f in db.list_factors())
    print(f"annotated {updated}/{before} records in {path}")
    print("required_tier distribution:", dict(tiers))


if __name__ == "__main__":
    main()
