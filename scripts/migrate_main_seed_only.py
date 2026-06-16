"""Make the *main* factor library strictly seed-only.

The canonical ``data/factors/factor_db.json`` historically mixed the ~88 seed
(formulaic) alphas with researcher factors.  Strict modularisation requires main
to hold ONLY the seed alphas; everything researched lives in a (config, prerun)
scope.  This one-shot migration:

1. backs up the current main DB (timestamped) — unless ``--no-backup``;
2. moves the RESEARCHER records (+ any trading ideas + a copy of their generated
   ``.py``) into a preserved ``legacy`` scope
   (``data/workspaces/legacy/preruns/main/``), so nothing is lost and they stay
   re-mergeable via ``run_merge.py``;
3. rewrites main = seed records only (and re-annotates capability tiers).

With ``--migrate-preruns`` it also copies any existing flat preruns
(``data/factors/preruns/<name>/`` + their read-logs + any
``data/preruns_downstream/<name>/`` books) into scopes under a config name.

Idempotent (re-running is a no-op once main is seed-only) and supports
``--dry-run``.

Usage::

    ./venv/bin/python scripts/migrate_main_seed_only.py --dry-run
    ./venv/bin/python scripts/migrate_main_seed_only.py
    ./venv/bin/python scripts/migrate_main_seed_only.py --migrate-preruns
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the package importable when run directly as ``scripts/...`` from the root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_fund_agent.databases import FactorDatabase
from quant_fund_agent.schemas import FactorSource
from quant_fund_agent.workspace import WORKSPACES_ROOT, Scope

log = logging.getLogger("migrate_main_seed_only")

MAIN_FACTOR_DB = Path("data/factors/factor_db.json")
PRERUNS_ROOT = Path("data/factors/preruns")
PAPER_READ_LOG_ROOT = Path("data/papers/preruns")
DOWNSTREAM_ROOT = Path("data/preruns_downstream")


def split_main_db(
    main_path: Path = MAIN_FACTOR_DB,
    legacy: Scope | None = None,
    *,
    backup: bool = True,
    dry_run: bool = False,
) -> dict:
    """Move researcher factors out of ``main_path`` into the ``legacy`` scope.

    Returns a report dict.  Idempotent: if main already holds no researcher
    records, nothing is moved.
    """
    legacy = legacy or Scope("legacy", "main")
    db = FactorDatabase()
    db.load_from_json(main_path)
    seeds = db.list_factors(source=FactorSource.SEED)
    researcher = db.list_factors(source=FactorSource.RESEARCHER)
    ideas = db.list_trading_ideas()

    report = {
        "main_path": str(main_path),
        "seeds": len(seeds),
        "researcher": len(researcher),
        "legacy_scope": legacy.label,
        "backup": None,
        "already_seed_only": not researcher,
        "dry_run": dry_run,
    }
    if not researcher:
        log.info("main is already seed-only (%d seeds) — nothing to move.", len(seeds))
        return report

    if backup and not dry_run:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = main_path.with_name(f"{main_path.stem}.backup-{ts}{main_path.suffix}")
        shutil.copy2(main_path, backup_path)
        report["backup"] = str(backup_path)
        log.info("backed up main → %s", backup_path)

    # ── legacy scope DB (idempotent: merge into any existing) ──
    legacy_db = FactorDatabase()
    if legacy.factor_db_path.exists():
        legacy_db.load_from_json(legacy.factor_db_path)
    for f in researcher:
        f.metadata.setdefault("provenance", {
            "scope": legacy.label, "config_hash": "unknown", "migrated_from": "main",
        })
        legacy_db.add_factor(f)
    for idea in ideas:
        legacy_db.add_trading_idea(idea)

    # ── seed-only main ──
    seed_db = FactorDatabase()
    for f in seeds:
        seed_db.add_factor(f)
    seed_db.annotate_tiers()

    if dry_run:
        log.info("[dry-run] would move %d researcher factors → %s; main → %d seeds.",
                 len(researcher), legacy.factor_db_path, len(seeds))
        return report

    legacy.ensure()
    # snapshot the researcher .py into the legacy scope (best-effort archival copy)
    for f in researcher:
        if f.code_path and Path(f.code_path).exists():
            try:
                shutil.copy2(f.code_path, legacy.factor_code_dir / Path(f.code_path).name)
            except Exception as e:  # pragma: no cover — archival copy is best-effort
                log.debug("could not snapshot %s: %s", f.code_path, e)
    legacy_db.save_to_json(legacy.factor_db_path)
    if not legacy.config_snapshot_path.exists():
        legacy.config_snapshot_path.write_text(json.dumps({
            "config_hash": "unknown", "config_name": legacy.config,
            "note": "migrated from main; original data config unknown",
        }, indent=2))
    legacy.write_manifest(kind="legacy-migration", n_factors=len(researcher))
    seed_db.save_to_json(main_path)

    log.info("moved %d researcher factors → %s; main now %d seeds.",
             len(researcher), legacy.factor_db_path, len(seeds))
    return report


def migrate_preruns(
    config_name: str,
    *,
    preruns_root: Path = PRERUNS_ROOT,
    papers_root: Path = PAPER_READ_LOG_ROOT,
    downstream_root: Path = DOWNSTREAM_ROOT,
    ws_root: Path = WORKSPACES_ROOT,
    config_data=None,
    dry_run: bool = False,
) -> list[str]:
    """Copy existing flat preruns into ``data/workspaces/<config_name>/preruns/``.

    Copies (not moves) so legacy paths keep working until callers are migrated.
    Returns the list of scope labels created.
    """
    migrated: list[str] = []
    if not preruns_root.exists():
        return migrated

    for d in sorted(p for p in preruns_root.iterdir() if p.is_dir()):
        if not (d / "factor_db.json").exists():
            continue
        scope = Scope(config_name, d.name, root=ws_root)
        migrated.append(scope.label)
        if dry_run:
            log.info("[dry-run] would migrate prerun %s → %s", d.name, scope.factor_db_path)
            continue
        scope.ensure()
        shutil.copy2(d / "factor_db.json", scope.factor_db_path)
        if (d / "manifest.json").exists():
            shutil.copy2(d / "manifest.json", scope.manifest_path)
        rl = papers_root / f"{d.name}_read_log.json"
        if rl.exists():
            shutil.copy2(rl, scope.read_log_path)
        ds = downstream_root / d.name
        if ds.exists():
            for src, dst in (("strategy_db.json", scope.strategy_db_path),
                             ("portfolio_db.json", scope.portfolio_db_path),
                             ("showcase.json", scope.showcase_path)):
                if (ds / src).exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ds / src, dst)
        # Flat preruns predate the config system; their original data config is
        # unknown (often the legacy LOBSTER sample), so DON'T assume the active
        # config — record it as unknown unless the caller asserts one.
        if config_data is not None:
            scope.write_config_snapshot(config_data)
        elif not scope.config_snapshot_path.exists():
            scope.config_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            scope.config_snapshot_path.write_text(json.dumps({
                "config_hash": "unknown", "config_name": scope.config,
                "note": "migrated legacy flat prerun; original data config unknown",
            }, indent=2))
        log.info("migrated prerun %s → %s", d.name, scope.factor_db_path)
    return migrated


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    p.add_argument("--no-backup", action="store_true",
                   help="Skip the timestamped backup of the main DB.")
    p.add_argument("--migrate-preruns", action="store_true",
                   help="Also copy existing flat preruns into config-scoped workspaces.")
    p.add_argument("--config-name", default=None,
                   help="Config scope for --migrate-preruns (default: 'legacy', "
                        "since flat preruns' original data config is unknown).")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    args = _parse_args()

    report = split_main_db(backup=not args.no_backup, dry_run=args.dry_run)
    print(json.dumps(report, indent=2))

    if args.migrate_preruns:
        # Default to the 'legacy' config: flat preruns predate the config system
        # and were typically researched on the legacy LOBSTER data, NOT whatever
        # config happens to be active now.  Pass --config-name to assert one.
        config_name = args.config_name or "legacy"
        labels = migrate_preruns(config_name, dry_run=args.dry_run)
        print(f"preruns migrated into config {config_name!r}: {labels or '(none)'}")


if __name__ == "__main__":
    main()
