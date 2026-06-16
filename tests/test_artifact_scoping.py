"""Tests for the workspace env seam on returns + model artifacts (Phase 1).

The fix makes both the fitted ``.joblib`` location and the per-strategy returns
dir resolvable from the environment (``MODEL_ARTIFACT_DIR`` /
``STRATEGY_RETURNS_DIR``), read *live* at call time — the same mechanism as
``FACTOR_DB_PATH``, so a scoped run keeps its artifacts/returns isolated, even
across the MCP modeling subprocess (which inherits ``os.environ``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.databases import StrategyDatabase
from quant_fund_agent.modeling.train import _artifact_dir, fit_and_backtest
from quant_fund_agent.schemas import StrategyRecord

N_BARS, N_TICKERS, HORIZON = 400, 6, 6
FACTOR_IDS = ["f1", "f2", "f3"]


@pytest.fixture
def panel():
    rng = np.random.default_rng(7)
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


def test_artifact_dir_reads_env(monkeypatch, tmp_path):
    monkeypatch.delenv("MODEL_ARTIFACT_DIR", raising=False)
    assert _artifact_dir().name == "models"  # default fallback
    monkeypatch.setenv("MODEL_ARTIFACT_DIR", str(tmp_path / "scoped_models"))
    assert _artifact_dir() == tmp_path / "scoped_models"


def test_fit_writes_joblib_to_env_dir(panel, monkeypatch, tmp_path):
    """With no explicit artifact_dir, the .joblib follows MODEL_ARTIFACT_DIR."""
    signals, data = panel
    scoped = tmp_path / "ws_models"
    monkeypatch.setenv("MODEL_ARTIFACT_DIR", str(scoped))

    out = fit_and_backtest(
        factor_signals_is=signals, data_is=data, model_type="ridge",
        factor_ids=FACTOR_IDS, target_horizon=HORIZON, strategy_id="arch_envseam",
    )  # note: no artifact_dir kwarg

    assert out["artifact_path"] == str(scoped / "arch_envseam.joblib")
    assert (scoped / "arch_envseam.joblib").exists()


def test_explicit_artifact_dir_overrides_env(panel, monkeypatch, tmp_path):
    signals, data = panel
    monkeypatch.setenv("MODEL_ARTIFACT_DIR", str(tmp_path / "env_dir"))
    explicit = tmp_path / "explicit_dir"
    out = fit_and_backtest(
        factor_signals_is=signals, data_is=data, model_type="ridge",
        factor_ids=FACTOR_IDS, target_horizon=HORIZON, strategy_id="arch_x",
        artifact_dir=explicit,
    )
    assert out["artifact_path"] == str(explicit / "arch_x.joblib")
    assert not (tmp_path / "env_dir" / "arch_x.joblib").exists()


def test_strategy_db_returns_dir_env_fallback(monkeypatch, tmp_path):
    scoped = tmp_path / "ws_returns"
    monkeypatch.setenv("STRATEGY_RETURNS_DIR", str(scoped))
    db = StrategyDatabase()  # no explicit returns_dir
    assert db._returns_dir == scoped

    db.add_strategy(StrategyRecord(id="strat_a", name="A"))
    s = pd.Series([0.001, -0.002, 0.003],
                  index=pd.date_range("2020-01-01", periods=3, freq="D"))
    path = db.save_returns("strat_a", s)
    assert path == scoped / "strat_a.csv"
    assert path.exists()


def test_strategy_db_explicit_returns_dir_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("STRATEGY_RETURNS_DIR", str(tmp_path / "env_returns"))
    explicit = tmp_path / "explicit_returns"
    db = StrategyDatabase(returns_dir=explicit)
    assert db._returns_dir == explicit
