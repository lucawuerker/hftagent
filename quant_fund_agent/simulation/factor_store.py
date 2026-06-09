"""Run-scoped factor bookkeeping for the walk-forward backtest.

A backtest must never touch the *permanent* factor library
(``data/factors/factor_db.json`` + the researcher ``.py`` files that ship in the
package).  Instead each run gets its **own** factor DB under its output folder,
seeded from the permanent library according to the run's ``factor_universe``:

- ``"all"``     → seed + every permanent researcher factor (the whole library);
- ``"session"`` → seed factors only, so the Selector sees seeds + only the
                  factors this run's research meetings discover.

Research meetings then persist into the run-scoped DB (the ``FACTOR_DB_PATH`` env
var points there for the duration of the run — see :mod:`simulation.simulator`).
The generated ``.py`` code still lands in the package ``factors/researcher/`` dir
(that is where ``discover_factors`` looks), so at the end of the run we
**snapshot** this run's generated code into the run folder and **purge** it from
the package — keeping the package clean while leaving the run's factors fully
recoverable.

These helpers live here (not in ``simulator.py``) to keep the driver lean and so
the seeding/snapshot logic can be unit-tested in isolation.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from quant_fund_agent.databases import FactorDatabase
from quant_fund_agent.schemas import FactorSource

log = logging.getLogger("simulation.factor_store")


def permanent_factor_ids(global_db_path: Path) -> set[str]:
    """Ids in the permanent factor DB, captured at the start of a run.

    Anything the run subsequently writes into its run-scoped DB whose id is NOT
    in this set is a *this-run* discovery — that is the set we snapshot + purge.
    """
    db = FactorDatabase()
    db.load_from_json(global_db_path)
    return {f.id for f in db.list_factors()}


def seed_run_factor_db(
    global_db_path: Path,
    run_db_path: Path,
    universe: str,
) -> int:
    """Seed a run-scoped factor DB from the permanent library.

    ``universe="all"`` copies every factor; ``universe="session"`` copies only
    the SEED factors so the run starts from the deterministic baseline and the
    Selector sees only seeds + this run's discoveries.  Returns the number of
    factors written.
    """
    src = FactorDatabase()
    src.load_from_json(global_db_path)

    out = FactorDatabase()
    if universe == "session":
        factors = src.list_factors(source=FactorSource.SEED)
    else:  # "all"
        factors = src.list_factors()
    for f in factors:
        out.add_factor(f)
    # Carry trading ideas across too (cheap, keeps the catalog self-consistent).
    for idea in src.list_trading_ideas():
        out.add_trading_idea(idea)

    run_db_path.parent.mkdir(parents=True, exist_ok=True)
    out.save_to_json(run_db_path)
    log.info("Seeded run factor DB (%s) with %d factors → %s",
             universe, len(factors), run_db_path)
    return len(factors)


def snapshot_and_purge_run_code(
    run_db_path: Path,
    permanent_ids: set[str],
    code_src_dir: Path,
    dest_dir: Path,
) -> list[str]:
    """Snapshot this run's researcher code into the run folder, purge the package.

    A "this-run" factor is any ``RESEARCHER`` factor in the run-scoped DB whose id
    is not in ``permanent_ids``.  For each, the ``<id>.py`` file written under
    ``code_src_dir`` (the package ``factors/researcher/`` dir) is copied to
    ``dest_dir`` and then removed from the package — so permanent factors' code is
    never touched and the package does not accumulate per-run files.

    Returns the list of factor ids that were snapshotted.
    """
    db = FactorDatabase()
    db.load_from_json(run_db_path)
    run_factors = [
        f for f in db.list_factors(source=FactorSource.RESEARCHER)
        if f.id not in permanent_ids
    ]
    if not run_factors:
        return []

    dest_dir.mkdir(parents=True, exist_ok=True)
    snapshotted: list[str] = []
    for f in run_factors:
        src = code_src_dir / f"{f.id}.py"
        if not src.exists():
            continue
        try:
            shutil.copy2(src, dest_dir / src.name)
            src.unlink()
            snapshotted.append(f.id)
        except Exception as e:  # pragma: no cover — best-effort cleanup
            log.warning("could not snapshot/purge run factor %s: %s", f.id, e)

    log.info("Snapshotted %d run-generated factor file(s) → %s (purged from package)",
             len(snapshotted), dest_dir)
    return snapshotted
