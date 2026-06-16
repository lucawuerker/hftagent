"""Two scopes never collide on returns / model artifacts / factor DB (Phase 2).

This is the regression guard for the bug that motivated the refactor: before,
``data/strategies/{returns,models}`` were global, so two prerun runs overwrote
each other's PnL CSVs and ``.joblib`` files.  Activating a :class:`Scope` now
routes both (plus the factor DB) into the scope's own tree via the env seam.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.databases import FactorDatabase, StrategyDatabase
from quant_fund_agent.modeling.train import fit_and_backtest
from quant_fund_agent.schemas import (
    FactorRecord,
    FactorSource,
    StrategyRecord,
    TradingIdeaCategory,
)
from quant_fund_agent.workspace import Scope, compose_factor_db

N_BARS, N_TICKERS, HORIZON = 400, 6, 6
FACTOR_IDS = ["f1", "f2", "f3"]

_SCOPE_ENV = ("FACTOR_DB_PATH", "PAPER_READ_LOG", "STRATEGY_RETURNS_DIR",
              "MODEL_ARTIFACT_DIR", "QF_SCOPE")


@pytest.fixture(autouse=True)
def _restore_scope_env():
    """``Scope.activate`` mutates os.environ directly — restore it after each test
    so the scope paths don't leak into other tests in the session."""
    saved = {k: __import__("os").environ.get(k) for k in _SCOPE_ENV}
    yield
    import os
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture
def panel():
    rng = np.random.default_rng(11)
    idx = pd.date_range("2020-01-01", periods=N_BARS, freq="10s")
    cols = [f"T{i}" for i in range(N_TICKERS)]
    signals = {
        fid: pd.DataFrame(rng.normal(size=(N_BARS, N_TICKERS)), index=idx, columns=cols)
        for fid in FACTOR_IDS
    }
    close = pd.DataFrame(
        100 + np.cumsum(rng.normal(0, 0.01, (N_BARS, N_TICKERS)), axis=0),
        index=idx, columns=cols,
    )
    return signals, {"close": close}


def _run_one_scope(scope: Scope, panel, sid: str):
    """Mimic the run_fund seam: activate → fit (artifact) → register returns."""
    scope.activate()
    signals, data = panel
    out = fit_and_backtest(
        factor_signals_is=signals, data_is=data, model_type="ridge",
        factor_ids=FACTOR_IDS, target_horizon=HORIZON, strategy_id=sid,
    )  # no artifact_dir → follows MODEL_ARTIFACT_DIR set by activate()

    db = StrategyDatabase()  # no returns_dir → follows STRATEGY_RETURNS_DIR
    db.add_strategy(StrategyRecord(id=sid, name=sid))
    returns = pd.Series(
        np.random.default_rng(0).normal(0, 1e-3, 20),
        index=pd.date_range("2020-02-01", periods=20, freq="D"),
    )
    db.save_returns(sid, returns)
    return out["artifact_path"], db._returns_dir


def test_two_scopes_disjoint_returns_and_artifacts(panel, tmp_path):
    a = Scope("yfinance_equity_demo", "modelA", root=tmp_path / "workspaces")
    b = Scope("yfinance_equity_demo", "modelB", root=tmp_path / "workspaces")

    art_a, ret_a = _run_one_scope(a, panel, "arch_a")
    art_b, ret_b = _run_one_scope(b, panel, "arch_b")

    # Artifacts landed under each scope's own models dir.
    assert art_a == str(a.models_dir / "arch_a.joblib")
    assert art_b == str(b.models_dir / "arch_b.joblib")
    assert (a.models_dir / "arch_a.joblib").exists()
    assert (b.models_dir / "arch_b.joblib").exists()
    # A's artifact is NOT in B's dir and vice-versa (the old collision).
    assert not (a.models_dir / "arch_b.joblib").exists()
    assert not (b.models_dir / "arch_a.joblib").exists()

    # Returns CSVs are isolated per scope.
    assert ret_a == a.returns_dir
    assert (a.returns_dir / "arch_a.csv").exists()
    assert (b.returns_dir / "arch_b.csv").exists()
    assert not (a.returns_dir / "arch_b.csv").exists()


def test_compose_uses_main_seeds_plus_scope_research(tmp_path, monkeypatch):
    main = tmp_path / "factors" / "factor_db.json"
    base = FactorDatabase()
    base.add_factor(FactorRecord(id="alpha_001", name="a1", source=FactorSource.SEED,
                                 category=TradingIdeaCategory.OTHER))
    base.save_to_json(main)
    monkeypatch.setattr("quant_fund_agent.workspace.MAIN_FACTOR_DB", main)

    scope = Scope("cfg", "pr", root=tmp_path / "ws").ensure()
    rdb = FactorDatabase()
    rdb.add_factor(FactorRecord(id="researched_x", name="x",
                                source=FactorSource.RESEARCHER,
                                category=TradingIdeaCategory.OTHER))
    rdb.save_to_json(scope.factor_db_path)

    n = compose_factor_db(scope.composed_db_path, scope.factor_db_path,
                          include_seeds=True)
    assert n == 2
    composed = FactorDatabase()
    composed.load_from_json(scope.composed_db_path)
    assert {f.id for f in composed.list_factors()} == {"alpha_001", "researched_x"}

    # --no-seeds → only the scope's researched factors.
    n2 = compose_factor_db(scope.composed_db_path, scope.factor_db_path,
                           include_seeds=False)
    assert n2 == 1
