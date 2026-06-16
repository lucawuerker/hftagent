"""preruns shim resolves scope-based preruns while keeping its API (Phase 6)."""

from __future__ import annotations

from quant_fund_agent.databases import FactorDatabase
from quant_fund_agent.factors import preruns
from quant_fund_agent.schemas import FactorRecord, FactorSource, TradingIdeaCategory
from quant_fund_agent.workspace import Scope


def _researcher_scope(root, config, prerun, fid):
    sc = Scope(config, prerun, root=root).ensure()
    db = FactorDatabase()
    db.add_factor(FactorRecord(id=fid, name=fid, source=FactorSource.RESEARCHER,
                               category=TradingIdeaCategory.OTHER))
    db.save_to_json(sc.factor_db_path)
    return sc


def test_db_path_resolves_scope(tmp_path, monkeypatch):
    sc = _researcher_scope(tmp_path / "ws", "cfg", "scopepr", "rx")
    monkeypatch.setattr(preruns, "list_scopes", lambda: [sc])

    # No legacy flat dir for "scopepr" → resolves to the scope's factor DB.
    assert preruns.db_path("scopepr") == sc.factor_db_path
    assert preruns.read_log_path("scopepr") == sc.read_log_path
    assert "scopepr" in preruns.list_preruns()


def test_build_downstream_factor_db_for_scope(tmp_path, monkeypatch):
    sc = _researcher_scope(tmp_path / "ws", "cfg", "scopepr", "rx")
    monkeypatch.setattr(preruns, "list_scopes", lambda: [sc])

    base = tmp_path / "main.json"
    bdb = FactorDatabase()
    bdb.add_factor(FactorRecord(id="alpha_001", name="seed", source=FactorSource.SEED,
                                category=TradingIdeaCategory.OTHER))
    bdb.save_to_json(base)

    target = tmp_path / "composed.json"
    n = preruns.build_downstream_factor_db(target, "scopepr", include_seeds=True,
                                           base_db=base)
    assert n == 2
    out = FactorDatabase()
    out.load_from_json(target)
    assert {f.id for f in out.list_factors()} == {"alpha_001", "rx"}
