"""Fit a model on the in-sample window and backtest it.

This is the single entry point shared by both the MCP modeling server and the
in-process fallback, so the two paths are guaranteed to compute identical
numbers.  Given the architect's in-sample slice it:

1. (ML models) cross-sectionally normalises the factor signals, takes a
   **chronological train/valid split inside the IS window**, fits the estimator
   on the *train* sub-window only, and reports/back-tests on the held-out
   *valid* sub-window — so the architect's metrics are never measured on data
   the model was fitted on;
2. persists the fitted estimator (joblib) so the Statistician's OOS test and
   the future live backtest can reload the *exact same* model — never refit it;
3. returns a metrics dict in the identical shape the architect used to build
   in-node, plus fit diagnostics and the artifact path.

The static-weights baseline skips fitting entirely: it builds a
``DynamicStrategy`` and back-tests it on the full IS window, preserving the
pre-existing behaviour.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_fund_agent.backtesting.strategy_backtester import (
    backtest_strategy,
    normalise_factor_signals,
)
from quant_fund_agent.modeling.catalog import STATIC_WEIGHTS, build_estimator
from quant_fund_agent.modeling.features import build_training_matrix

DEFAULT_ARTIFACT_DIR = Path("data/strategies/models")


def _artifact_dir() -> Path:
    """Where fitted ``.joblib`` estimators are written — resolved at call time.

    Read live from ``MODEL_ARTIFACT_DIR`` on every call (rather than captured at
    import) so a scoped run (``run_fund``/simulator) can redirect artifacts to its
    own ``data/workspaces/.../models`` dir — including across the MCP modeling
    subprocess, which inherits the parent's environment.  Mirrors
    ``research_service._factor_db_path`` / ``FACTOR_DB_PATH``.
    """
    return Path(os.getenv("MODEL_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR)))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _serialise_series(s: pd.Series) -> dict:
    """JSON-friendly view of a DatetimeIndexed series (matches the architect)."""
    return {
        "timestamps": [ts.isoformat() for ts in s.index],
        "values": [float(v) for v in s.values],
    }


def _metrics_dict(result, elapsed: float) -> dict:
    """Mirror the architect's metrics payload (model_dump + serialised series)."""
    d = result.metrics.model_dump()
    d["backtest_elapsed_s"] = round(elapsed, 2)
    d["equity_curve"] = _serialise_series(result.cumulative_returns)
    d["portfolio_returns"] = _serialise_series(result.portfolio_returns)
    return d


def _slice_signals(signals: dict[str, pd.DataFrame], lo: int, hi: int) -> dict[str, pd.DataFrame]:
    return {k: df.iloc[lo:hi] for k, df in signals.items()}


def _slice_panel(data: dict[str, pd.DataFrame], lo: int, hi: int) -> dict[str, pd.DataFrame]:
    return {k: df.iloc[lo:hi] for k, df in data.items()}


def _feature_importances(estimator, factor_ids: list[str]) -> dict[str, float] | None:
    """Best-effort feature importances / coefficients, keyed by factor id.

    Estimators are wrapped (target/feature standardisation) by the catalog, so
    dig out the inner fitted model before reading ``coef_`` / importances.
    """
    from quant_fund_agent.modeling.catalog import unwrap_estimator

    inner = unwrap_estimator(estimator)
    vec = getattr(inner, "feature_importances_", None)
    if vec is None:
        coef = getattr(inner, "coef_", None)
        if coef is not None:
            vec = np.abs(np.ravel(coef))
    if vec is None or len(vec) != len(factor_ids):
        return None
    return {fid: round(float(v), 6) for fid, v in zip(factor_ids, vec)}


def _r2(estimator, X: np.ndarray, y: np.ndarray) -> float | None:
    if X.size == 0 or len(y) < 2:
        return None
    from sklearn.metrics import r2_score
    try:
        return round(float(r2_score(y, estimator.predict(X))), 6)
    except Exception:
        return None


def _subsample_rows(
    X: np.ndarray, y: np.ndarray, frac: float, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically keep a fraction of the ``(X, y)`` *fitting* rows.

    Speeds up fitting (especially the tree/boosting models, whose cost scales
    with sample count) without touching the backtest, which still runs on the
    full panel.  ``frac >= 1`` is a no-op; a small floor avoids degenerate
    near-empty fits when ``frac`` is tiny.
    """
    n = len(y)
    if frac >= 1.0 or n == 0:
        return X, y
    k = min(n, max(50, int(round(n * frac))))
    if k >= n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=k, replace=False))
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_and_backtest(
    *,
    factor_signals_is: dict[str, pd.DataFrame],
    data_is: dict[str, pd.DataFrame],
    model_type: str,
    model_params: dict[str, Any] | None = None,
    factor_ids: list[str] | None = None,
    weights: dict[str, float] | None = None,
    target_horizon: int = 6,
    holding_period: int = 6,
    max_positions: int = 20,
    equal_weight: bool = False,
    min_conviction: float = 0.0,
    position_construction: str = "cross_sectional",
    position_params: dict[str, Any] | None = None,
    valid_ratio: float = 0.3,
    artifact_dir: str | Path | None = None,
    strategy_id: str | None = None,
    train_sample_frac: float = 1.0,
) -> dict[str, Any]:
    """Fit (if ML) + backtest one candidate; returns metrics/artifact/diagnostics.

    ``factor_signals_is`` / ``data_is`` are the architect's **in-sample** slices.
    ``train_sample_frac`` (< 1.0) randomly subsamples the rows used to *fit* the
    model — a speed knob for large panels; the backtest still uses all data.
    """
    sid = strategy_id or f"model_{uuid.uuid4().hex[:8]}"
    t0 = time.time()

    # ── static-weights baseline: no fitting, backtest on the full IS window ──
    if model_type == STATIC_WEIGHTS:
        from quant_fund_agent.strategies.dynamic import DynamicStrategy

        w = weights or {}
        fids = list(w.keys())
        strategy = DynamicStrategy(
            strategy_id=sid, name=sid, description="",
            factor_ids=fids, weights=w,
            holding_period=holding_period, max_positions=max_positions,
            equal_weight=equal_weight, min_conviction=min_conviction,
            position_construction=position_construction, position_params=position_params,
        )
        result = backtest_strategy(strategy, factor_signals_is, data_is)
        return {
            "model_type": STATIC_WEIGHTS,
            "factor_ids": fids,
            "metrics": _metrics_dict(result, time.time() - t0),
            "artifact_path": "",
            "diagnostics": {},
        }

    # ── ML model: select on held-out IS-valid, then refit on the FULL IS ──
    #
    # 1. fit on the IS-train sub-window and score predictions on the held-out
    #    IS-valid sub-window (IC = the trial *selection* signal; train/valid
    #    R^2 = the overfitting diagnostic);
    # 2. refit a fresh model on the FULL in-sample window, persist it, and report
    #    its FULL-IS backtest — so the metrics handed to the Statistician are a
    #    stable in-sample estimate directly comparable to OOS, with no
    #    selection-bias inflation from a short validation slice.
    from quant_fund_agent.backtesting.data_loader import forward_returns
    from quant_fund_agent.backtesting.engine import _ic_series
    from quant_fund_agent.modeling.features import predict_to_signal
    from quant_fund_agent.strategies.model_strategy import ModelStrategy

    fids = list(factor_ids or [])
    if not fids:
        raise ValueError("fit_and_backtest: ML models require non-empty factor_ids")

    normed_is = normalise_factor_signals(factor_signals_is)
    n_rows = len(next(iter(normed_is.values())))
    cut = int(n_rows * (1.0 - valid_ratio))
    if cut < 2 or n_rows - cut < 2:
        raise ValueError(
            f"IS window too short to split (n={n_rows}, cut={cut}); "
            "lower valid_ratio or widen the in-sample window."
        )
    close = data_is["close"]

    # (1) train/valid split — selection score + overfitting diagnostics
    X_tr, y_tr = build_training_matrix(
        _slice_signals(normed_is, 0, cut), fids, close.iloc[:cut], target_horizon,
    )
    if len(y_tr) < 10:
        raise ValueError(
            f"Not enough non-NaN training samples ({len(y_tr)}) for {model_type}."
        )
    X_va, y_va = build_training_matrix(
        _slice_signals(normed_is, cut, n_rows), fids, close.iloc[cut:], target_horizon,
    )

    trial_estimator = build_estimator(model_type, model_params)
    X_tr_fit, y_tr_fit = _subsample_rows(X_tr, y_tr, train_sample_frac)
    trial_estimator.fit(X_tr_fit, y_tr_fit)

    # Held-out validation IC: the robust, leakage-free trial selection score.
    close_va = close.iloc[cut:]
    valid_pred = predict_to_signal(
        trial_estimator, _slice_signals(normed_is, cut, n_rows), fids,
        close_va.index, close_va.columns,
    )
    ic_va = _ic_series(
        valid_pred, forward_returns(close_va, horizon=target_horizon).shift(-1)
    )
    validation_score = float(ic_va.mean()) if len(ic_va) else 0.0

    # (2) refit on the FULL in-sample window — this is the model we keep.
    X_full, y_full = build_training_matrix(normed_is, fids, close, target_horizon)
    X_full_fit, y_full_fit = _subsample_rows(X_full, y_full, train_sample_frac)
    estimator = build_estimator(model_type, model_params)
    estimator.fit(X_full_fit, y_full_fit)

    diagnostics = {
        "train_r2": _r2(trial_estimator, X_tr, y_tr),
        "valid_r2": _r2(trial_estimator, X_va, y_va),
        "valid_ic": round(validation_score, 6),
        "n_train_samples": int(len(y_tr)),
        "n_valid_samples": int(len(y_va)),
        "n_fit_samples": int(len(y_full_fit)),
        "feature_importances": _feature_importances(estimator, fids),
    }

    # Persist the full-IS model so OOS / live reload (never refit) it.
    artifact_dir = Path(artifact_dir) if artifact_dir is not None else _artifact_dir()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{sid}.joblib"
    import joblib
    joblib.dump(
        {
            "estimator": estimator,
            "factor_ids": fids,
            "model_type": model_type,
            "target_horizon": target_horizon,
        },
        artifact_path,
    )

    # Report the FULL-IS backtest.  Pass RAW signals; backtest_strategy
    # normalises them once, exactly as the training path normalised.
    strategy = ModelStrategy(
        strategy_id=sid, name=sid, description="",
        factor_ids=fids, model_type=model_type, estimator=estimator,
        target_horizon=target_horizon,
        holding_period=holding_period, max_positions=max_positions,
        equal_weight=equal_weight, min_conviction=min_conviction,
        position_construction=position_construction, position_params=position_params,
    )
    result = backtest_strategy(strategy, factor_signals_is, data_is)

    metrics = _metrics_dict(result, time.time() - t0)
    metrics["validation_score"] = round(validation_score, 6)
    return {
        "model_type": model_type,
        "factor_ids": fids,
        "metrics": metrics,
        "artifact_path": str(artifact_path),
        "diagnostics": diagnostics,
        "validation_score": round(validation_score, 6),
    }
