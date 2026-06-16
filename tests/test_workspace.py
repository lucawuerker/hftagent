"""Tests for the (config, prerun) layout abstraction (Phase 0).

Covers the on-disk path derivation for :class:`Scope` / :class:`Book`, the env
seam keys, factor-DB composition (seeds + researcher), config fingerprint/name
helpers, and the config-snapshot mismatch warning.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from quant_fund_agent.config import (
    DataSettings,
    config_fingerprint,
    default_config_name,
    provenance_stamp,
)
from quant_fund_agent.databases import FactorDatabase
from quant_fund_agent.schemas import FactorRecord, FactorSource, TradingIdeaCategory
from quant_fund_agent.workspace import Book, Scope, list_books, list_scopes


def test_scope_paths_are_nested_under_config_and_prerun():
    s = Scope("yfinance_equity_demo", "gpt4omini", root=Path("data/workspaces"))
    assert s.config_dir == Path("data/workspaces/yfinance_equity_demo")
    assert s.dir == Path("data/workspaces/yfinance_equity_demo/preruns/gpt4omini")
    assert s.config_snapshot_path == s.config_dir / "config.snapshot.json"
    assert s.factor_db_path == s.dir / "factors" / "factor_db.json"
    assert s.composed_db_path == s.dir / "factors" / "_composed.json"
    assert s.factor_code_dir == s.dir / "factors" / "code"
    assert s.strategy_db_path == s.dir / "strategies" / "strategy_db.json"
    assert s.returns_dir == s.dir / "strategies" / "returns"
    assert s.models_dir == s.dir / "strategies" / "models"
    assert s.portfolio_db_path == s.dir / "portfolio" / "portfolio_db.json"
    assert s.read_log_path == s.dir / "papers_read.json"
    assert s.label == "yfinance_equity_demo/gpt4omini"


def test_scope_export_env_keys():
    s = Scope("cfg", "pr")
    env = s.export_env()
    assert set(env) == {
        "FACTOR_DB_PATH", "PAPER_READ_LOG",
        "STRATEGY_RETURNS_DIR", "MODEL_ARTIFACT_DIR", "QF_SCOPE",
    }
    assert env["QF_SCOPE"] == "cfg/pr"
    assert env["MODEL_ARTIFACT_DIR"].endswith("preruns/pr/strategies/models")


def test_book_paths_and_env():
    b = Book("active", root=Path("data/books"))
    assert b.dir == Path("data/books/active")
    assert b.factor_db_path == b.dir / "factors" / "factor_db.json"
    assert b.models_dir == b.dir / "strategies" / "models"
    env = b.export_env()
    assert env["QF_SCOPE"] == "book:active"
    assert "PAPER_READ_LOG" not in env


def _seed(fid: str) -> FactorRecord:
    return FactorRecord(id=fid, name=fid, source=FactorSource.SEED,
                        category=TradingIdeaCategory.OTHER)


def _researcher(fid: str) -> FactorRecord:
    return FactorRecord(id=fid, name=fid, source=FactorSource.RESEARCHER,
                        category=TradingIdeaCategory.OTHER)


def test_compose_runtime_db_merges_seeds_and_researcher(tmp_path):
    main = tmp_path / "factors" / "factor_db.json"
    base = FactorDatabase()
    base.add_factor(_seed("alpha_001"))
    base.add_factor(_seed("alpha_002"))
    base.save_to_json(main)

    s = Scope("cfg", "pr", root=tmp_path / "workspaces")
    s.ensure()
    rdb = FactorDatabase()
    rdb.add_factor(_researcher("order_flow"))
    rdb.save_to_json(s.factor_db_path)

    # Patch the module-level main DB pointer for composition.
    import quant_fund_agent.workspace as wsmod
    old = wsmod.MAIN_FACTOR_DB
    wsmod.MAIN_FACTOR_DB = main
    try:
        n = s.compose_runtime_db(include_seeds=True)
    finally:
        wsmod.MAIN_FACTOR_DB = old

    assert n == 3
    composed = FactorDatabase()
    composed.load_from_json(s.composed_db_path)
    ids = {f.id for f in composed.list_factors()}
    assert ids == {"alpha_001", "alpha_002", "order_flow"}


def test_config_fingerprint_stable_and_sensitive():
    a = DataSettings(provider="yfinance", asset_class="equity",
                     universe_preset="demo", start="2024-01-01", end="2026-01-01")
    b = dataclasses.replace(a)
    assert config_fingerprint(a) == config_fingerprint(b)
    # Changing a panel-defining dimension changes the hash.
    assert config_fingerprint(a) != config_fingerprint(
        dataclasses.replace(a, provider="fmp"))
    assert config_fingerprint(a) != config_fingerprint(
        dataclasses.replace(a, start="2020-01-01"))
    # A non-panel field (reporting lag) must NOT change it.
    assert config_fingerprint(a) == config_fingerprint(
        dataclasses.replace(a, reporting_lag_days=999))


def test_default_config_name():
    d = DataSettings(provider="yfinance", asset_class="equity", universe_preset="demo")
    assert default_config_name(d) == "yfinance_equity_demo"
    d2 = DataSettings(provider="fmp", asset_class="crypto", universe_preset=None,
                      tickers=["BTC-USD", "ETH-USD"])
    assert default_config_name(d2) == "fmp_crypto_custom2"


def test_provenance_stamp_carries_config():
    d = DataSettings(provider="yfinance", asset_class="equity", universe_preset="demo")
    stamp = provenance_stamp("yfinance_equity_demo/gpt4omini", d)
    assert stamp["scope"] == "yfinance_equity_demo/gpt4omini"
    assert stamp["provider"] == "yfinance"
    assert stamp["config_hash"] == config_fingerprint(d)


def test_config_snapshot_mismatch_warning(tmp_path):
    s = Scope("cfg", "pr", root=tmp_path)
    d1 = DataSettings(provider="yfinance", asset_class="equity", universe_preset="demo")
    s.write_config_snapshot(d1)
    assert s.config_mismatch(d1) is None  # same config → no warning
    d2 = dataclasses.replace(d1, provider="fmp")
    msg = s.config_mismatch(d2)
    assert msg is not None
    assert "may have been researched" in msg
    # Snapshot is write-once: a second write under a different config keeps the first.
    s.write_config_snapshot(d2)
    snap = json.loads(s.config_snapshot_path.read_text())
    assert snap["config_hash"] == config_fingerprint(d1)


def test_list_scopes_and_books(tmp_path):
    s = Scope("cfg_a", "pr1", root=tmp_path / "ws").ensure()
    FactorDatabase().save_to_json(s.factor_db_path)
    Scope("cfg_a", "pr2", root=tmp_path / "ws").ensure()
    (tmp_path / "books" / "active").mkdir(parents=True)

    scopes = list_scopes(root=tmp_path / "ws")
    assert {(x.config, x.prerun) for x in scopes} == {("cfg_a", "pr1"), ("cfg_a", "pr2")}
    books = list_books(root=tmp_path / "books")
    assert [b.name for b in books] == ["active"]
