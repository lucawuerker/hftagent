"""Backtesting engine — IC / ICIR computation for single factors.

The core metric is the **Information Coefficient (IC)**: the
cross-sectional Pearson correlation between the factor signal
at time *t* and the realised forward return over the next *horizon*
bars.  This is computed for every timestamp, giving an IC time series.

From the IC series we derive:
    IC mean    – average predictive power
    IC std     – stability of the signal
    IC IR      – IC mean / IC std  (information ratio of the IC series)
    IC hit rate – fraction of timestamps where IC > 0
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.backtesting.data_loader import forward_returns
from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.schemas import BacktestMetrics
from quant_fund_agent.strategies.base import BaseStrategy


def _ic_series(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    min_assets: int = 5,
) -> pd.Series:
    """Cross-sectional Pearson IC per timestamp (vectorised).

    Demeans each row, then computes Pearson correlation row-by-row using the
    ``cov(x,y) / (std(x)*std(y))`` formula — fully vectorised with no Python-level
    row loop.
    """
    common_idx = signal.index.intersection(fwd_ret.index)
    common_cols = signal.columns.intersection(fwd_ret.columns)
    sig = signal.loc[common_idx, common_cols]
    ret = fwd_ret.loc[common_idx, common_cols]

    valid = sig.notna() & ret.notna()
    n_valid = valid.sum(axis=1)

    sig = sig.where(valid)
    ret = ret.where(valid)

    sig_mean = sig.mean(axis=1)
    ret_mean = ret.mean(axis=1)

    sig_dm = sig.sub(sig_mean, axis=0)
    ret_dm = ret.sub(ret_mean, axis=0)

    cov = (sig_dm * ret_dm).sum(axis=1)
    sig_std = (sig_dm ** 2).sum(axis=1).pow(0.5)
    ret_std = (ret_dm ** 2).sum(axis=1).pow(0.5)

    denom = sig_std * ret_std
    ic = cov / denom.replace(0, np.nan)

    ic = ic[n_valid >= min_assets].dropna()
    ic.name = "IC"
    return ic


def backtest_factor(
    factor: BaseFactor | None,
    data: dict[str, pd.DataFrame],
    horizon: int = 6,
    horizons: tuple[int, ...] = (1, 6, 60),
    *,
    signal: pd.DataFrame | None = None,
    factor_horizon: int | None = None,
) -> BacktestMetrics:
    """Compute IC-based metrics for a single factor.

    Args:
        factor: instantiated factor with a ``.calc()`` method. May be ``None``
            when ``signal`` is supplied directly (e.g. from a cache).
        data: OHLCV panel dict (index=timestamps, columns=tickers).
        horizon: legacy single default horizon (kept for compatibility).
        horizons: set of forward-return horizons in bars to evaluate.
        signal: pre-computed factor signal; skips ``factor.calc(data)`` when provided.
        factor_horizon: the factor's *own* prediction horizon.  When given, it is
            measured (unioned into ``horizons``) and the headline IC + the
            ``extra.horizon_bars`` are anchored at it instead of ``horizon``.
            ``None`` reproduces the legacy behaviour byte-for-byte.

    Returns:
        BacktestMetrics with IC, IC std, ICIR, IC hit rate populated.
    """
    if signal is None:
        assert factor is not None, "factor required when signal is not provided"
        signal = factor.calc(data)
    ic_by_horizon: dict[str, dict[str, float | int | None]] = {}

    eval_horizons = set(horizons)
    if factor_horizon is not None:
        eval_horizons.add(int(factor_horizon))
    for h in sorted(eval_horizons):
        fwd_ret_h = forward_returns(data["close"], horizon=h)
        ic_h = _ic_series(signal, fwd_ret_h)
        if ic_h.empty:
            ic_by_horizon[str(h)] = {
                "ic": None,
                "ic_std": None,
                "ic_ir": None,
                "ic_hit_rate": None,
                "n_timestamps": 0,
            }
            continue

        ic_mean = float(ic_h.mean())
        ic_std = float(ic_h.std())
        ic_ir = float(ic_mean / ic_std) if ic_std > 0 else 0.0
        ic_hit = float((ic_h > 0).mean())
        ic_by_horizon[str(h)] = {
            "ic": round(ic_mean, 6),
            "ic_std": round(ic_std, 6),
            "ic_ir": round(ic_ir, 6),
            "ic_hit_rate": round(ic_hit, 4),
            "n_timestamps": int(len(ic_h)),
        }

    # Anchor the top-level metrics at the factor's own horizon when supplied,
    # else the legacy default horizon (for byte-identical back-compat).
    if factor_horizon is not None:
        default_h = str(int(factor_horizon))
    else:
        default_h = str(horizon if horizon in set(horizons) else sorted(set(horizons))[0])
    default_metrics = ic_by_horizon.get(default_h, {})

    return BacktestMetrics(
        information_coefficient=default_metrics.get("ic"),
        ic_std=default_metrics.get("ic_std"),
        ic_ir=default_metrics.get("ic_ir"),
        ic_hit_rate=default_metrics.get("ic_hit_rate"),
        ic_by_horizon=ic_by_horizon,
        extra={
            "n_timestamps": default_metrics.get("n_timestamps", 0),
            "horizon_bars": int(default_h),
            "horizons_bars": sorted(eval_horizons),
        },
    )
