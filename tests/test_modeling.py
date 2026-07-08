"""Unit tests for the modelling toolbox (pure synthetic data, no MCP)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.backtesting.data_loader import forward_returns
from quant_fund_agent.backtesting.strategy_backtester import backtest_strategy
from quant_fund_agent.modeling import (
    FITTABLE_MODEL_TYPES,
    STATIC_WEIGHTS,
    available_model_types,
    build_estimator,
    fit_and_backtest,
    is_model_available,
    list_model_menu,
)
from quant_fund_agent.modeling.features import build_training_matrix, predict_to_signal
from quant_fund_agent.strategies.model_strategy import ModelStrategy

N_BARS = 600
N_TICKERS = 8
FACTOR_IDS = ["f1", "f2", "f3"]
HORIZON = 6

# Smaller ensembles keep the test suite fast.
FAST_PARAMS = {
    "random_forest": {"n_estimators": 30, "max_depth": 4},
    "gradient_boosting": {"n_estimators": 30, "max_depth": 2},
    "xgboost": {"n_estimators": 30, "max_depth": 3},
    "lightgbm": {"n_estimators": 30},
}


@pytest.fixture
def panel():
    """Synthetic IS slice: factor signals + an OHLC 'close' frame."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2020-01-01", periods=N_BARS, freq="10s")
    cols = [f"T{i}" for i in range(N_TICKERS)]

    signals = {
        fid: pd.DataFrame(rng.normal(size=(N_BARS, N_TICKERS)), index=idx, columns=cols)
        for fid in FACTOR_IDS
    }
    # Make returns mildly predictable from the factors so models can learn.
    close = pd.DataFrame(
        100 + np.cumsum(rng.normal(0, 0.01, (N_BARS, N_TICKERS)), axis=0),
        index=idx, columns=cols,
    )
    return signals, {"close": close}


def test_build_training_matrix_no_lookahead(panel):
    signals, data = panel
    X, y = build_training_matrix(signals, FACTOR_IDS, data["close"], target_horizon=HORIZON)

    # Features align with target, no NaNs leak through.
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == len(FACTOR_IDS)
    assert np.isfinite(X).all() and np.isfinite(y).all()

    # The target is the *forward* return, so the final ``HORIZON`` bars have no
    # label and must be excluded — guarantees row t never sees its own future.
    expected = (N_BARS - HORIZON) * N_TICKERS
    assert X.shape[0] == expected


def test_predict_to_signal_shape(panel):
    signals, data = panel
    X, y = build_training_matrix(signals, FACTOR_IDS, data["close"], target_horizon=HORIZON)
    est = build_estimator("ridge", {})
    est.fit(X, y)

    sig = predict_to_signal(est, signals, FACTOR_IDS, data["close"].index, data["close"].columns)
    assert sig.shape == data["close"].shape
    assert list(sig.columns) == list(data["close"].columns)
    # Most rows are predictable (only the all-NaN-feature rows, of which there
    # are none here, would be NaN).
    assert np.isfinite(sig.values).mean() > 0.9


def test_build_training_matrix_imputes_sparse_factor():
    """A sparse factor must not collapse the joint sample set (regression).

    Previously ``dropna(how="any")`` kept only rows where *every* factor was
    non-NaN, so one sparse factor among many wiped out training/prediction.  The
    fix imputes missing features to 0 and keeps any sample with a finite target
    and at least one present feature.
    """
    rng = np.random.default_rng(3)
    idx = pd.date_range("2020-01-01", periods=N_BARS, freq="10s")
    cols = [f"T{i}" for i in range(N_TICKERS)]
    dense = pd.DataFrame(rng.normal(size=(N_BARS, N_TICKERS)), index=idx, columns=cols)
    sparse = dense.copy()
    sparse.iloc[:, :] = np.nan
    sparse.iloc[::50, 0] = 1.0  # present for ~2% of cells only
    signals = {"dense": dense, "sparse": sparse}
    close = pd.DataFrame(
        100 + np.cumsum(rng.normal(0, 0.01, (N_BARS, N_TICKERS)), axis=0),
        index=idx, columns=cols,
    )

    X, y = build_training_matrix(signals, ["dense", "sparse"], close, target_horizon=HORIZON)
    # Driven by the dense factor: nearly every labelled cell survives (vs. the
    # ~12 the old complete-case logic would have kept).
    assert X.shape[0] == (N_BARS - HORIZON) * N_TICKERS
    assert np.isfinite(X).all()  # NaNs were imputed, not left in
    # The sparse column is mostly the neutral 0 fill.
    assert (X[:, 1] == 0.0).mean() > 0.9

    est = build_estimator("ridge", {})
    est.fit(X, y)
    sig = predict_to_signal(est, signals, ["dense", "sparse"], close.index, close.columns)
    # Predictions populate wherever the dense factor is present (was ~all-NaN).
    assert np.isfinite(sig.values).mean() > 0.9


def test_build_training_matrix_caps_samples():
    """Fitting rows are bounded so the CV linear models stay fast on big panels.

    The cap is a deterministic uniform subsample of the kept rows; below the cap
    every row is kept (no effect on small windows).
    """
    rng = np.random.default_rng(11)
    idx = pd.date_range("2020-01-01", periods=N_BARS, freq="10s")
    cols = [f"T{i}" for i in range(N_TICKERS)]
    signals = {f: pd.DataFrame(rng.normal(size=(N_BARS, N_TICKERS)), index=idx, columns=cols)
               for f in FACTOR_IDS}
    close = pd.DataFrame(
        100 + np.cumsum(rng.normal(0, 0.01, (N_BARS, N_TICKERS)), axis=0),
        index=idx, columns=cols,
    )
    full = (N_BARS - HORIZON) * N_TICKERS

    # No cap (large default) → every labelled row kept.
    X0, y0 = build_training_matrix(signals, FACTOR_IDS, close, HORIZON)
    assert X0.shape[0] == full == y0.shape[0]

    # Explicit small cap → exactly that many rows, deterministically.
    cap = 100
    X1, y1 = build_training_matrix(signals, FACTOR_IDS, close, HORIZON, max_samples=cap)
    X2, y2 = build_training_matrix(signals, FACTOR_IDS, close, HORIZON, max_samples=cap)
    assert X1.shape[0] == cap == y1.shape[0]
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)  # reproducible


def test_unavailable_models_excluded_from_menu():
    """Models whose native lib is missing must drop out of the menu cleanly."""
    menu_types = {m["model_type"] for m in list_model_menu()}
    for mt in FITTABLE_MODEL_TYPES:
        assert (mt in menu_types) == is_model_available(mt)
    # The pure scikit-learn suite is always available.
    for mt in ("linear_regression", "ridge", "lasso", "elastic_net",
               "random_forest", "gradient_boosting"):
        assert is_model_available(mt) and mt in menu_types


def test_subsample_rows_deterministic_and_floored():
    from quant_fund_agent.modeling.train import _subsample_rows

    X = np.arange(2000 * 3, dtype=float).reshape(2000, 3)
    y = np.arange(2000, dtype=float)

    # frac >= 1 is a no-op (identity)
    X1, y1 = _subsample_rows(X, y, 1.0)
    assert X1.shape == X.shape and np.array_equal(y1, y)

    # frac=0.1 keeps ~10% of rows, sorted, and is deterministic
    Xa, ya = _subsample_rows(X, y, 0.1)
    Xb, yb = _subsample_rows(X, y, 0.1)
    assert len(ya) == 200
    assert np.array_equal(ya, yb)               # deterministic
    assert np.all(np.diff(ya) > 0)              # sorted, unique
    assert np.array_equal(Xa[:, 0], ya * 3)     # rows stay aligned to X

    # tiny frac is floored (avoids degenerate near-empty fits)
    _, yc = _subsample_rows(X, y, 0.0001)
    assert len(yc) == 50


def test_fit_and_backtest_train_sample_frac(panel, tmp_path):
    """train_sample_frac fits on fewer rows but still backtests on all data."""
    signals, data = panel
    full = fit_and_backtest(
        factor_signals_is=signals, data_is=data, model_type="ridge",
        factor_ids=FACTOR_IDS, target_horizon=HORIZON, artifact_dir=tmp_path,
        strategy_id="frac_full", train_sample_frac=1.0,
    )
    sub = fit_and_backtest(
        factor_signals_is=signals, data_is=data, model_type="ridge",
        factor_ids=FACTOR_IDS, target_horizon=HORIZON, artifact_dir=tmp_path,
        strategy_id="frac_sub", train_sample_frac=0.1,
    )
    # Far fewer rows actually fitted, yet a real (finite) backtest still runs.
    assert sub["diagnostics"]["n_fit_samples"] < full["diagnostics"]["n_fit_samples"] / 5
    assert np.isfinite(sub["metrics"]["sharpe_ratio"])


@pytest.mark.parametrize("model_type", available_model_types(include_static=False))
def test_fit_and_backtest_each_model(panel, tmp_path, model_type):
    signals, data = panel
    result = fit_and_backtest(
        factor_signals_is=signals,
        data_is=data,
        model_type=model_type,
        model_params=FAST_PARAMS.get(model_type, {}),
        factor_ids=FACTOR_IDS,
        target_horizon=HORIZON,
        artifact_dir=tmp_path,
        strategy_id=f"test_{model_type}",
    )

    assert result["model_type"] == model_type
    assert result["factor_ids"] == FACTOR_IDS

    metrics = result["metrics"]
    assert "sharpe_ratio" in metrics
    assert "portfolio_returns" in metrics and metrics["portfolio_returns"]["values"]

    diag = result["diagnostics"]
    assert diag["n_train_samples"] > 0 and diag["n_valid_samples"] > 0
    assert "train_r2" in diag and "valid_r2" in diag
    # held-out validation IC drives trial selection
    assert "valid_ic" in diag
    assert "validation_score" in result and "validation_score" in metrics

    # A loadable artifact must exist for OOS / live reuse.
    artifact = result["artifact_path"]
    assert artifact and (tmp_path / f"test_{model_type}.joblib").exists()


def test_static_weights_path(panel, tmp_path):
    signals, data = panel
    result = fit_and_backtest(
        factor_signals_is=signals,
        data_is=data,
        model_type=STATIC_WEIGHTS,
        weights={fid: 1.0 for fid in FACTOR_IDS},
        artifact_dir=tmp_path,
        strategy_id="test_static",
    )
    assert result["model_type"] == STATIC_WEIGHTS
    assert result["artifact_path"] == ""  # static models are not persisted
    assert "sharpe_ratio" in result["metrics"]


def test_model_strategy_artifact_roundtrip(panel, tmp_path):
    signals, data = panel
    result = fit_and_backtest(
        factor_signals_is=signals,
        data_is=data,
        model_type="ridge",
        factor_ids=FACTOR_IDS,
        target_horizon=HORIZON,
        artifact_dir=tmp_path,
        strategy_id="roundtrip",
    )

    strategy = ModelStrategy.from_artifact(
        result["artifact_path"], strategy_id="roundtrip_oos",
    )
    assert strategy.factor_ids == FACTOR_IDS
    assert strategy.target_horizon == HORIZON

    # The reloaded model runs cleanly through the standard backtester.
    bt = backtest_strategy(strategy, signals, data)
    assert bt.metrics.sharpe_ratio is not None
    assert len(bt.portfolio_returns) > 0


def test_bad_params_are_clamped_not_crashed(panel, tmp_path):
    """An absurd hyper-parameter from the LLM must be clamped, not fatal."""
    signals, data = panel
    result = fit_and_backtest(
        factor_signals_is=signals,
        data_is=data,
        model_type="ridge",
        model_params={"alpha": "not-a-number"},  # coerced to default
        factor_ids=FACTOR_IDS,
        target_horizon=HORIZON,
        artifact_dir=tmp_path,
        strategy_id="clamp",
    )
    assert "sharpe_ratio" in result["metrics"]
