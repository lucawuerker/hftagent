"""Factor + strategy records carry a (config, prerun) provenance stamp (Phase 3).

Every researched factor and built strategy records which scope + data config it
was born under, so a record stays auditable after it is pooled into a book.
"""

from __future__ import annotations

from quant_fund_agent.agents.architect.state import StrategySpec
from quant_fund_agent.config import config_fingerprint, get_settings
from quant_fund_agent.databases import FactorDatabase
from quant_fund_agent.pipeline import (
    StrategyPipelineResult,
    _scope_provenance,
    build_strategy_record,
)
from quant_fund_agent.schemas import FactorRecord, FactorSource, TradingIdeaCategory


def test_factor_provenance_stamped_on_persist(tmp_path, monkeypatch):
    from quant_fund_agent.mcp import research_service

    db_path = tmp_path / "factor_db.json"
    monkeypatch.setenv("FACTOR_DB_PATH", str(db_path))
    monkeypatch.setenv("QF_SCOPE", "yfinance_equity_demo/gpt4omini")

    rec = FactorRecord(id="researched_z", name="z", source=FactorSource.RESEARCHER,
                       category=TradingIdeaCategory.OTHER)
    research_service.persist_results([rec.model_dump(mode="json")], [])

    db = FactorDatabase()
    db.load_from_json(db_path)
    prov = db.get_factor("researched_z").metadata["provenance"]
    assert prov["scope"] == "yfinance_equity_demo/gpt4omini"
    assert prov["config_hash"] == config_fingerprint(get_settings().data)
    assert prov["provider"] == get_settings().data.provider


def test_scope_provenance_defaults_to_main(monkeypatch):
    monkeypatch.delenv("QF_SCOPE", raising=False)
    assert _scope_provenance()["scope"] == "main"
    monkeypatch.setenv("QF_SCOPE", "cfg/pr")
    assert _scope_provenance()["scope"] == "cfg/pr"


def test_strategy_provenance_stamped(monkeypatch):
    monkeypatch.setenv("QF_SCOPE", "fmp_crypto_demo/claude")
    spec = StrategySpec(
        strategy_name="Test Strat", model_type="static_weights",
        weights={"alpha_001": 1.0}, factor_ids=["alpha_001"],
    )
    res = StrategyPipelineResult(
        selector_result={"hypothesis": "h"},
        architect_result={"strategy_spec": spec, "backtest_metrics": {},
                          "trial_history": []},
        stat_result=None,
        approved=True,
    )
    record, _ = build_strategy_record(res, strategy_id="strat_prov")
    prov = record.metadata["provenance"]
    assert prov["scope"] == "fmp_crypto_demo/claude"
    assert prov["config_hash"] == config_fingerprint(get_settings().data)
