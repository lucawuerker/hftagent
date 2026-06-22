"""Track — ML-combined signal → per-underlying vectorised backtest (no agents).

Each catalog model is fed *all* of a prerun's researched factors and learns to
**combine** them into a single predicted signal: it fits ``factors → forward
return`` on the in-sample window and predicts one value per (timestamp,
underlying).  That combined signal is then run through the *simple* per-underlying
vectorised backtest (:mod:`comparison.vector_backtest`) — ``position(signal) × the
underlying's own forward return`` — on the held-out out-of-sample tail.

This isolates "how good is this zoo as raw features for a return model?" from the
downstream agents, and — unlike the old cross-sectional long-short backtest — it
treats each factor combination as a standalone directional signal, so it needs no
cross-section (works with any number of underlyings).

Factors are standardised **per underlying over time** before fitting (using
in-sample statistics, so no out-of-sample leakage); ``fit_standardize`` switches
back to the cross-sectional z-score.  ``±inf`` from a factor (e.g. a ratio with a
near-zero denominator) is treated as missing — otherwise it poisons the mean/std
and the model fit.  The equal-weight **ensemble** averages the models' (rescaled)
combined signals and backtests that.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

from quant_fund_agent.comparison.vector_backtest import vector_backtest

log = logging.getLogger("comparison.bruteforce")


def _standardised_features(
    factor_ids: list[str], panel: dict[str, Any], is_idx, cfg,
) -> dict[str, pd.DataFrame]:
    """Each factor's signal, standardised for fitting (per-underlying or cross-sectional).

    ``±inf`` is replaced with NaN first so it neither warns in the mean/std nor
    leaks into the fit (it is later imputed to 0 = the per-underlying mean).
    """
    from quant_fund_agent.backtesting.strategy_backtester import normalise_factor_signals
    from quant_fund_agent.modeling.service import _factor_signal

    close = panel["close"]
    feats: dict[str, pd.DataFrame] = {}
    for fid in factor_ids:
        sig = (_factor_signal(fid, panel)
               .reindex(index=close.index, columns=close.columns)
               .replace([np.inf, -np.inf], np.nan))
        if cfg.fit_standardize == "cross_sectional":
            feats[fid] = normalise_factor_signals({fid: sig})[fid]
        else:  # per-underlying over time, using IS-window stats (no OOS leakage)
            mu = sig.loc[is_idx].mean()
            sd = sig.loc[is_idx].std().replace(0, np.nan)
            feats[fid] = (sig - mu) / sd
    return feats


def _pooled_features(factor_ids: list[str], panel: dict[str, Any], cfg):
    """Build the shared, sanitised ``(T·N) × F`` matrix + target + IS train mask.

    Computed once per prerun (not per model).  Row order is timestamp-major so a
    prediction vector reshapes straight back to a ``(T×N)`` combined signal.
    """
    from quant_fund_agent.backtesting.data_loader import forward_returns

    close = panel["close"]
    index, cols = close.index, close.columns
    n_rows, n_cols = len(index), len(cols)
    cut = int(n_rows * (1.0 - cfg.oos_split_ratio))

    feats = _standardised_features(factor_ids, panel, index[:cut], cfg)
    X = np.column_stack([feats[fid].to_numpy(dtype=float).ravel() for fid in factor_ids])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)  # missing/∞ → the (0) mean
    y = forward_returns(close, horizon=cfg.target_horizon).to_numpy(dtype=float).ravel()

    is_rows = np.arange(n_rows * n_cols) < (cut * n_cols)
    train = np.flatnonzero(is_rows & np.isfinite(y))
    if cfg.train_sample_frac < 1.0 and len(train) > 100:
        rng = np.random.default_rng(cfg.seed)
        train = np.sort(rng.choice(train, size=max(100, int(len(train) * cfg.train_sample_frac)),
                                   replace=False))
    return X, y, train, index, cols, n_cols


def evaluate_prerun_models(
    prerun: str, factor_ids: list[str], panel: dict[str, Any], cfg,
) -> list[dict[str, Any]]:
    """One row per catalog model (+ ensemble): vectorised backtest of its combined signal."""
    from quant_fund_agent.modeling.catalog import build_estimator

    rows: list[dict[str, Any]] = []
    models = cfg.resolved_models()
    log.info("ML-combine: prerun '%s' — %d factors → %d models (fit_standardize=%s, "
             "position=%s, aggregation=%s)", prerun, len(factor_ids), len(models),
             cfg.fit_standardize, cfg.position_mode, cfg.backtest_aggregation)

    X, y, train, index, cols, n_cols = _pooled_features(factor_ids, panel, cfg)
    if len(train) < 50:
        log.warning("ML-combine '%s': only %d IS training rows — skipping.", prerun, len(train))
        return [{"prerun": prerun, "model": m, "n_factors_used": len(factor_ids),
                 "error": "insufficient IS rows"} for m in models]

    combined: dict[str, pd.DataFrame] = {}
    with warnings.catch_warnings():
        # Benign sklearn deprecation from LassoCV/ElasticNetCV (n_alphas → alphas).
        warnings.filterwarnings("ignore", category=FutureWarning)
        # LightGBM auto-names numpy columns ("Column_0"…) at fit, so predicting on a
        # nameless array trips sklearn's feature-name check — cosmetic, predictions
        # are identical.  Filter just that message.
        warnings.filterwarnings("ignore", message="X does not have valid feature names",
                                category=UserWarning)
        for model in models:
            log.info("ML-combine: prerun '%s' fitting '%s' → combined signal ...", prerun, model)
            try:
                est = build_estimator(model, cfg.fast_model_params(model))
                est.fit(X[train], y[train])
                pred = np.asarray(est.predict(X), dtype=float).reshape(len(index), n_cols)
                sig = pd.DataFrame(pred, index=index, columns=cols)
                metrics = vector_backtest(sig, panel, cfg)
            except Exception as e:  # noqa: BLE001 — one model must not abort the sweep
                log.warning("ML-combine %s/%s failed: %s", prerun, model, e)
                rows.append({"prerun": prerun, "model": model,
                             "n_factors_used": len(factor_ids), "error": str(e)[:200]})
                continue
            combined[model] = sig
            rows.append({"prerun": prerun, "model": model,
                         "n_factors_used": len(factor_ids), **metrics})

    if cfg.include_ensemble and len(combined) >= 2:
        log.info("ML-combine: prerun '%s' building ensemble of %d models ...", prerun, len(combined))
        # Rescale each combined signal to comparable spread, then average.
        parts = [s / (float(s.stack().std()) or 1.0) for s in combined.values()]
        ens = sum(parts) / len(parts)
        try:
            metrics = vector_backtest(ens, panel, cfg)
            rows.append({"prerun": prerun, "model": "ensemble", "n_factors_used": len(factor_ids),
                         **metrics, "ensemble_members": ",".join(combined)})
        except Exception as e:  # noqa: BLE001
            log.warning("ML-combine ensemble failed for %s: %s", prerun, e)
    return rows
