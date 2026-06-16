"""Main seed-only migration moves researcher factors to a legacy scope (Phase 4).

Verifies the split is correct (main = seeds, legacy = researcher), a backup is
made, nothing is lost, and a re-run is idempotent.
"""

from __future__ import annotations

import importlib

from quant_fund_agent.databases import FactorDatabase
from quant_fund_agent.schemas import FactorRecord, FactorSource, TradingIdeaCategory
from quant_fund_agent.workspace import Scope

migrate = importlib.import_module("scripts.migrate_main_seed_only")


def _make_main(path, n_seed=5, n_researcher=3):
    db = FactorDatabase()
    for i in range(n_seed):
        db.add_factor(FactorRecord(id=f"alpha_{i:03d}", name=f"a{i}",
                                   source=FactorSource.SEED,
                                   category=TradingIdeaCategory.OTHER))
    for i in range(n_researcher):
        db.add_factor(FactorRecord(id=f"researched_{i}", name=f"r{i}",
                                   source=FactorSource.RESEARCHER,
                                   category=TradingIdeaCategory.OTHER))
    db.save_to_json(path)
    return db


def test_split_main_to_seed_only(tmp_path):
    main = tmp_path / "factors" / "factor_db.json"
    _make_main(main, n_seed=5, n_researcher=3)
    legacy = Scope("legacy", "main", root=tmp_path / "workspaces")

    report = migrate.split_main_db(main, legacy, backup=True, dry_run=False)
    assert report["seeds"] == 5 and report["researcher"] == 3
    assert report["backup"] is not None

    # main is now seed-only
    main_db = FactorDatabase()
    main_db.load_from_json(main)
    assert len(main_db.list_factors(source=FactorSource.RESEARCHER)) == 0
    assert len(main_db.list_factors(source=FactorSource.SEED)) == 5

    # legacy holds the researcher factors, stamped + with a config snapshot
    legacy_db = FactorDatabase()
    legacy_db.load_from_json(legacy.factor_db_path)
    rids = {f.id for f in legacy_db.list_factors(source=FactorSource.RESEARCHER)}
    assert rids == {"researched_0", "researched_1", "researched_2"}
    f0 = legacy_db.get_factor("researched_0")
    assert f0.metadata["provenance"]["migrated_from"] == "main"
    assert legacy.config_snapshot_path.exists()

    # backup preserves the original (8 records)
    backups = list((tmp_path / "factors").glob("factor_db.backup-*.json"))
    assert len(backups) == 1
    backup_db = FactorDatabase()
    backup_db.load_from_json(backups[0])
    assert len(backup_db.list_factors()) == 8


def test_idempotent_rerun(tmp_path):
    main = tmp_path / "factors" / "factor_db.json"
    _make_main(main, n_seed=4, n_researcher=2)
    legacy = Scope("legacy", "main", root=tmp_path / "workspaces")

    migrate.split_main_db(main, legacy, backup=False, dry_run=False)
    # second pass: main already seed-only → no-op, no duplication
    report2 = migrate.split_main_db(main, legacy, backup=False, dry_run=False)
    assert report2["already_seed_only"] is True

    legacy_db = FactorDatabase()
    legacy_db.load_from_json(legacy.factor_db_path)
    assert len(legacy_db.list_factors(source=FactorSource.RESEARCHER)) == 2


def test_dry_run_writes_nothing(tmp_path):
    main = tmp_path / "factors" / "factor_db.json"
    _make_main(main, n_seed=3, n_researcher=2)
    legacy = Scope("legacy", "main", root=tmp_path / "workspaces")

    migrate.split_main_db(main, legacy, backup=True, dry_run=True)
    # main untouched, no legacy written, no backup
    main_db = FactorDatabase()
    main_db.load_from_json(main)
    assert len(main_db.list_factors(source=FactorSource.RESEARCHER)) == 2
    assert not legacy.factor_db_path.exists()
    assert not list((tmp_path / "factors").glob("factor_db.backup-*.json"))


def test_migrate_preruns_copies_into_scopes(tmp_path):
    preruns_root = tmp_path / "preruns"
    (preruns_root / "modelA").mkdir(parents=True)
    db = FactorDatabase()
    db.add_factor(FactorRecord(id="rx", name="rx", source=FactorSource.RESEARCHER,
                               category=TradingIdeaCategory.OTHER))
    db.save_to_json(preruns_root / "modelA" / "factor_db.json")
    (preruns_root / "modelA" / "manifest.json").write_text('{"llm_model": "x"}')

    labels = migrate.migrate_preruns(
        "yfinance_equity_demo",
        preruns_root=preruns_root,
        papers_root=tmp_path / "papers",
        downstream_root=tmp_path / "downstream",
        ws_root=tmp_path / "workspaces",
        dry_run=False,
    )
    assert labels == ["yfinance_equity_demo/modelA"]
    scope = Scope("yfinance_equity_demo", "modelA", root=tmp_path / "workspaces")
    assert scope.factor_db_path.exists()
    assert scope.manifest_path.exists()
