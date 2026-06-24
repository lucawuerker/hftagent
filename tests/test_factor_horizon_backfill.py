"""Backfill of ``prediction_horizon`` onto existing factor DB records.

Covers the constant strategy (the user's choice), the data-driven pick from the
measured ``ic_by_horizon`` grid, idempotency via the metadata sentinel, and the
``suggested_horizons`` fill.
"""

from __future__ import annotations

from quant_fund_agent.databases import FactorDatabase
from quant_fund_agent.schemas import (
    BacktestMetrics,
    FactorRecord,
    FactorSource,
    TradingIdeaCategory,
)


def _rec(fid: str, ic_by_horizon: dict | None = None) -> FactorRecord:
    bm = BacktestMetrics(ic_by_horizon=ic_by_horizon or {}) if ic_by_horizon is not None else None
    return FactorRecord(id=fid, name=fid, category=TradingIdeaCategory.MOMENTUM,
                        source=FactorSource.RESEARCHER, backtest_metrics=bm)


def _db(*recs: FactorRecord) -> FactorDatabase:
    db = FactorDatabase()
    for r in recs:
        db.add_factor(r)
    return db


def test_constant_backfill_sets_value_and_sentinel():
    db = _db(_rec("a"), _rec("b"))
    n = db.backfill_prediction_horizons(strategy="constant", value=6)
    assert n == 2
    for fid in ("a", "b"):
        rec = db.get_factor(fid)
        assert rec.prediction_horizon == 6
        assert rec.metadata.get("horizon_backfilled") == "constant"


def test_idempotent_without_force():
    db = _db(_rec("a"))
    assert db.backfill_prediction_horizons(strategy="constant") == 1
    # second pass is a no-op (sentinel present)
    assert db.backfill_prediction_horizons(strategy="constant") == 0
    # force re-stamps
    assert db.backfill_prediction_horizons(strategy="constant", force=True) == 1


def test_data_driven_picks_best_ic_ir():
    ibh = {
        "1": {"ic": 0.01, "ic_ir": 0.10},
        "6": {"ic": 0.03, "ic_ir": 0.50},   # strongest IR → chosen
        "60": {"ic": 0.05, "ic_ir": 0.20},
    }
    db = _db(_rec("a", ibh))
    db.backfill_prediction_horizons(strategy="data_driven", value=6)
    assert db.get_factor("a").prediction_horizon == 6


def test_data_driven_falls_back_when_no_metrics():
    db = _db(_rec("a", {}), _rec("b"))  # empty grid / no metrics
    db.backfill_prediction_horizons(strategy="data_driven", value=9)
    assert db.get_factor("a").prediction_horizon == 9
    assert db.get_factor("b").prediction_horizon == 9


def test_fill_suggested_uses_measured_grid():
    ibh = {"1": {"ic_ir": 0.1}, "6": {"ic_ir": 0.2}, "60": {"ic_ir": 0.05}}
    db = _db(_rec("a", ibh))
    db.backfill_prediction_horizons(strategy="constant", fill_suggested=True)
    assert db.get_factor("a").suggested_horizons == [1, 6, 60]
