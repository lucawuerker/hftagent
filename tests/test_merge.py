"""Merge pools scopes into a book; main stays untouched (Phase 5)."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_fund_agent.databases import FactorDatabase, StrategyDatabase
from quant_fund_agent.merge import merge_into_book
from quant_fund_agent.schemas import (
    FactorRecord,
    FactorSource,
    StrategyRecord,
    TradingIdeaCategory,
)
from quant_fund_agent.workspace import Book, Scope


@pytest.fixture
def main_seed_db(tmp_path, monkeypatch):
    """A seed-only main DB (one seed alpha) wired into merge.MAIN_FACTOR_DB."""
    main = tmp_path / "factors" / "factor_db.json"
    db = FactorDatabase()
    db.add_factor(FactorRecord(id="alpha_001", name="seed", source=FactorSource.SEED,
                               category=TradingIdeaCategory.OTHER))
    db.save_to_json(main)
    monkeypatch.setattr("quant_fund_agent.merge.MAIN_FACTOR_DB", main)
    return main


def _make_scope(root, config, prerun, *, factor_id, strategy_id=None,
                strategy_factors=None, config_hash="h1", with_artifact=True):
    sc = Scope(config, prerun, root=root).ensure()
    fdb = FactorDatabase()
    fdb.add_factor(FactorRecord(id=factor_id, name=factor_id,
                                source=FactorSource.RESEARCHER,
                                category=TradingIdeaCategory.OTHER,
                                metadata={"provenance": {"config_hash": config_hash}}))
    fdb.save_to_json(sc.factor_db_path)

    if strategy_id:
        artifact = ""
        if with_artifact:
            (sc.models_dir).mkdir(parents=True, exist_ok=True)
            art = sc.models_dir / f"{strategy_id}.joblib"
            art.write_bytes(b"dummy-estimator")
            artifact = str(art)
        sdb = StrategyDatabase(returns_dir=sc.returns_dir)
        rec = StrategyRecord(
            id=strategy_id, name=strategy_id,
            factor_ids=strategy_factors or [factor_id],
            model_params={"model_artifact_path": artifact},
            metadata={"provenance": {"config_hash": config_hash}},
        )
        sdb.add_strategy(rec)
        returns = pd.Series([0.001, -0.002, 0.003],
                            index=pd.date_range("2020-01-01", periods=3, freq="D"))
        sdb.save_returns(strategy_id, returns)
        sdb.flush_returns()
        sdb.save_to_json(sc.strategy_db_path)
    return sc


def test_merge_factors_keeps_provenance_and_main_untouched(main_seed_db, tmp_path):
    sc = _make_scope(tmp_path / "ws", "cfg", "prA", factor_id="researched_a")
    book = Book("active", root=tmp_path / "books")

    report = merge_into_book([sc], book, include_strategies=False)
    assert "researched_a" in report.promoted_factor_ids

    book_db = FactorDatabase()
    book_db.load_from_json(book.factor_db_path)
    ids = {f.id for f in book_db.list_factors()}
    assert ids == {"alpha_001", "researched_a"}  # seed + merged researcher
    merged = book_db.get_factor("researched_a")
    assert merged.source == FactorSource.RESEARCHER  # auditable, not relabelled seed
    assert merged.metadata["merged"]["from_scope"] == "cfg/prA"

    # main DB is byte-for-byte unchanged (still seed-only, one factor).
    main_db = FactorDatabase()
    main_db.load_from_json(main_seed_db)
    assert {f.id for f in main_db.list_factors()} == {"alpha_001"}


def test_merge_is_idempotent(main_seed_db, tmp_path):
    sc = _make_scope(tmp_path / "ws", "cfg", "prA", factor_id="researched_a")
    book = Book("active", root=tmp_path / "books")
    merge_into_book([sc], book, include_strategies=False)
    report2 = merge_into_book([sc], book, include_strategies=False)
    assert report2.promoted_factor_ids == []
    assert "researched_a" in report2.skipped_factor_ids


def test_merge_strategies_copies_artifacts_and_returns(main_seed_db, tmp_path):
    sc = _make_scope(tmp_path / "ws", "cfg", "prA", factor_id="researched_a",
                     strategy_id="strat_a", strategy_factors=["researched_a"])
    book = Book("active", root=tmp_path / "books")

    report = merge_into_book([sc], book, include_strategies=True)
    assert "strat_a" in report.promoted_strategy_ids
    # backing researched factor auto-included
    assert "researched_a" in report.promoted_factor_ids

    # artifact + returns copied into the book and paths rewritten
    assert (book.models_dir / "strat_a.joblib").exists()
    assert (book.returns_dir / "strat_a.csv").exists()
    book_sdb = StrategyDatabase(returns_dir=book.returns_dir)
    book_sdb.load_from_json(book.strategy_db_path)
    rec = book_sdb.get_strategy("strat_a")
    assert rec.model_params["model_artifact_path"] == str(book.models_dir / "strat_a.joblib")
    assert rec.returns_path == str(book.returns_dir / "strat_a.csv")


def test_merge_missing_backing_factor_fails_clearly(main_seed_db, tmp_path):
    # Strategy references a factor that exists in NO source and is not a seed.
    sc = _make_scope(tmp_path / "ws", "cfg", "prA", factor_id="researched_a",
                     strategy_id="strat_x", strategy_factors=["ghost_factor"])
    book = Book("active", root=tmp_path / "books")
    with pytest.raises(ValueError, match="ghost_factor"):
        merge_into_book([sc], book, include_strategies=True)
    # nothing written on failure
    assert not book.factor_db_path.exists()


def test_merge_cross_config_warns(main_seed_db, tmp_path):
    a = _make_scope(tmp_path / "ws", "cfg_a", "pr", factor_id="fa",
                    strategy_id="sa", strategy_factors=["alpha_001"], config_hash="hA")
    b = _make_scope(tmp_path / "ws", "cfg_b", "pr", factor_id="fb",
                    strategy_id="sb", strategy_factors=["alpha_001"], config_hash="hB")
    book = Book("active", root=tmp_path / "books")
    report = merge_into_book([a, b], book, include_strategies=True)
    assert {"sa", "sb"} <= set(report.promoted_strategy_ids)
    assert report.config_hash_warnings  # differing config hashes flagged
