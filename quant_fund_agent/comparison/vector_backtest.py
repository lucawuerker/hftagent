"""Simple per-underlying vectorised backtest of a signal.

A factor zoo's *combined* signal (an ML model's per-(timestamp, underlying)
prediction — see :mod:`comparison.bruteforce`) is evaluated **not** by ranking the
cross-section, but as a standalone directional bet on each underlying:

1. standardise the signal per underlying over time → ``z``;
2. map ``z`` to a position (long / flat / short);
3. P&L per underlying = ``position[t] · forward_return(t→t+h)`` — ``position[t]`` is
   known at ``t`` so there is no look-ahead;
4. aggregate across underlyings into one book (or report per-underlying);
5. summarise with the shared frequency-aware metrics, separately on IS and OOS.

Every modelling choice is a :class:`ComparisonConfig` knob (``position_mode``,
``position_threshold``, ``position_zscore_basis``, ``backtest_aggregation``).  This
needs no cross-section, so it works with any number of underlyings — even one.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from quant_fund_agent.backtesting.data_loader import forward_returns
from quant_fund_agent.backtesting.positions import (
    directional_positions,
    zscore_over_time,
)
from quant_fund_agent.backtesting.strategy_backtester import _compute_metrics

log = logging.getLogger("comparison.vector_backtest")


def _zscore(frame: pd.DataFrame, basis: str, window: int) -> pd.DataFrame:
    """Per-underlying standardisation over time (shared with the deployed fund)."""
    return zscore_over_time(frame, basis, window)


def _positions(z: pd.DataFrame, cfg) -> pd.DataFrame:
    """Map a (z-scored) signal to raw directional positions per the config's rule."""
    return directional_positions(z, mode=cfg.position_mode, threshold=cfg.position_threshold)


def _slice_metrics(pnl: pd.DataFrame, pos: pd.DataFrame, sig: pd.DataFrame,
                   close: pd.DataFrame, cfg) -> dict[str, Any]:
    """Summary metrics for one (IS or OOS) time-slice, honouring the aggregation."""
    h = cfg.target_horizon
    # Degenerate signal (e.g. LassoCV zeroed every coefficient → constant prediction,
    # or the band is never crossed) → no positions taken.  Report None, not a
    # misleading 0.0 Sharpe.
    if float(np.abs(pos.to_numpy(dtype=float)).sum()) == 0.0:
        return {}
    if cfg.backtest_aggregation == "per_underlying":
        sharpes, ann, dd, hit = [], [], [], []
        for c in pnl.columns:
            port = pnl[c].dropna()
            if len(port) < 2:
                continue
            m = _compute_metrics(port, pos[[c]], sig[[c]], close[[c]], ic_horizon=h)
            if m.sharpe_ratio is not None:
                sharpes.append(m.sharpe_ratio)
            if m.annualised_return is not None:
                ann.append(m.annualised_return)
            if m.max_drawdown is not None:
                dd.append(m.max_drawdown)
            if m.hit_rate is not None:
                hit.append(m.hit_rate)
        _m = lambda xs: float(np.mean(xs)) if xs else None  # noqa: E731
        return {"sharpe": _m(sharpes), "sharpe_std": float(np.std(sharpes)) if sharpes else None,
                "ann_return": _m(ann), "max_drawdown": _m(dd), "hit_rate": _m(hit), "ic_mean": None}

    # portfolio (default): equal-weight book — mean of per-underlying P&L each bar.
    port = pnl.mean(axis=1).dropna()
    if len(port) < 2:
        return {}
    m = _compute_metrics(port, pos, sig, close, ic_horizon=h)
    return {"sharpe": m.sharpe_ratio, "ann_return": m.annualised_return,
            "max_drawdown": m.max_drawdown, "hit_rate": m.hit_rate,
            "ic_mean": m.ic_mean, "turnover": m.avg_daily_turnover}


def vector_backtest(signal: pd.DataFrame, panel: dict[str, Any], cfg,
                    oos_split_ratio: float | None = None) -> dict[str, Any]:
    """Backtest a signal frame; return IS/OOS KPIs in the brute-force row schema."""
    close = panel["close"]
    sig = signal.reindex(index=close.index, columns=close.columns).replace([np.inf, -np.inf], np.nan)
    z = _zscore(sig, cfg.position_zscore_basis, cfg.position_zscore_window)
    pos = _positions(z, cfg)
    pnl = pos * forward_returns(close, horizon=cfg.target_horizon)

    # IS/OOS split: an explicit ``oos_split_ratio`` override keeps the contiguous
    # tail split; otherwise defer to the config (calendar windows if set, else the
    # ratio tail) so every track splits the panel identically.
    if oos_split_ratio is not None:
        cut = int(len(close.index) * (1.0 - oos_split_ratio))
        is_mask = np.zeros(len(close.index), dtype=bool)
        is_mask[:cut] = True
        oos_mask = ~is_mask
    else:
        is_mask, oos_mask = cfg.split_masks(close.index)
    is_m = _slice_metrics(pnl[is_mask], pos[is_mask], sig[is_mask], close[is_mask], cfg)
    oos_m = _slice_metrics(pnl[oos_mask], pos[oos_mask], sig[oos_mask], close[oos_mask], cfg)

    return {
        "is_ic": is_m.get("ic_mean"), "oos_ic": oos_m.get("ic_mean"),
        "is_sharpe": is_m.get("sharpe"), "oos_sharpe": oos_m.get("sharpe"),
        "is_ann_return": is_m.get("ann_return"), "oos_ann_return": oos_m.get("ann_return"),
        "oos_max_drawdown": oos_m.get("max_drawdown"), "oos_hit_rate": oos_m.get("hit_rate"),
        "oos_sharpe_std": oos_m.get("sharpe_std"),  # per_underlying aggregation only
    }
