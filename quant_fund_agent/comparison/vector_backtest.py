"""Simple per-underlying vectorised backtest of a signal.

A factor zoo's *combined* signal (an ML model's per-(timestamp, underlying)
prediction — see :mod:`comparison.bruteforce`) is evaluated **not** by ranking the
cross-section, but as a standalone directional bet on each underlying:

1. standardise the signal per underlying over time → ``z``;
2. map ``z`` to a target position (long / flat / short);
3. hold each bar's target for ``holding_period`` bars by layering ``1/holding_period``
   of capital in per bar — a staggered "tranche" book whose live position is the
   rolling mean of the last ``holding_period`` targets — and **mark to market on the
   1-bar forward return**: P&L per underlying = ``book[t] · forward_return(t→t+1)``.
   ``book[t]`` is known at ``t`` so there is no look-ahead, and because consecutive
   1-bar returns don't overlap, the Sharpe annualisation is unbiased.  (Multiplying a
   raw position by an ``h``-bar forward return — the previous approach — overlaps
   ``h−1`` bars between adjacent rows, which inflates the annualised return ~``h``×
   and the Sharpe ~``√h``×.)
4. aggregate across underlyings into one book (or report per-underlying);
5. summarise with the shared frequency-aware metrics, separately on IS and OOS.

Every modelling choice is a :class:`ComparisonConfig` knob (``holding_period``,
``position_mode``, ``position_threshold``, ``position_zscore_basis``,
``backtest_aggregation``).  This needs no cross-section, so it works with any number
of underlyings — even one.
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


def _per_underlying_ic(sig: pd.DataFrame, close: pd.DataFrame, h: int) -> float | None:
    """Pooled per-underlying IC of the combined signal vs its h-bar forward return.

    Non-cross-sectional (matches the IC track): standardise the signal per
    underlying over time, concatenate across underlyings, and Spearman-correlate
    with the forward return — so it is defined for a single ticker.
    """
    from quant_fund_agent.comparison.ic import _spearman
    from quant_fund_agent.comparison.standardize import per_underlying_zscore

    x = per_underlying_zscore(sig).to_numpy(dtype=float).ravel()
    y = forward_returns(close, horizon=h).to_numpy(dtype=float).ravel()
    ic, _ = _spearman(x, y)
    return ic


def _slice_metrics(pnl: pd.DataFrame, book: pd.DataFrame, sig: pd.DataFrame,
                   close: pd.DataFrame, cfg) -> dict[str, Any]:
    """Summary metrics for one (IS or OOS) time-slice, honouring the aggregation.

    ``book`` is the live (tranche-averaged) position whose 1-bar P&L is ``pnl``; it
    drives the turnover / position-count stats and the degenerate-signal guard.  The
    IC is measured *per underlying* (pooled) at the *forecast* horizon
    (``cfg.target_horizon``), which is independent of how long each tranche is held.
    """
    h = cfg.target_horizon
    # Degenerate signal (e.g. LassoCV zeroed every coefficient → constant prediction,
    # or the band is never crossed) → no positions taken.  Report None, not a
    # misleading 0.0 Sharpe.
    if float(np.abs(book.to_numpy(dtype=float)).sum()) == 0.0:
        return {}
    ic_mean = _per_underlying_ic(sig, close, h)
    if cfg.backtest_aggregation == "per_underlying":
        sharpes, ann, dd, hit = [], [], [], []
        for c in pnl.columns:
            port = pnl[c].dropna()
            if len(port) < 2:
                continue
            m = _compute_metrics(port, book[[c]], sig[[c]], close[[c]], ic_horizon=h)
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
                "ann_return": _m(ann), "max_drawdown": _m(dd), "hit_rate": _m(hit),
                "ic_mean": ic_mean}

    # portfolio (default): equal-weight book — mean of per-underlying P&L each bar.
    port = pnl.mean(axis=1).dropna()
    if len(port) < 2:
        return {}
    m = _compute_metrics(port, book, sig, close, ic_horizon=h)
    return {"sharpe": m.sharpe_ratio, "ann_return": m.annualised_return,
            "max_drawdown": m.max_drawdown, "hit_rate": m.hit_rate,
            "ic_mean": ic_mean, "turnover": m.avg_daily_turnover}


def vector_backtest(signal: pd.DataFrame, panel: dict[str, Any], cfg,
                    oos_split_ratio: float | None = None) -> dict[str, Any]:
    """Backtest a signal frame; return IS/OOS KPIs in the brute-force row schema."""
    close = panel["close"]
    sig = signal.reindex(index=close.index, columns=close.columns).replace([np.inf, -np.inf], np.nan)
    z = _zscore(sig, cfg.position_zscore_basis, cfg.position_zscore_window)
    pos = _positions(z, cfg)

    # Staggered "tranche" book: hold each bar's target for ``hold`` bars by layering
    # ``1/hold`` of capital in per bar → the live position is the rolling mean of the
    # last ``hold`` targets.  Marking that to market on the *1-bar* forward return
    # gives a non-overlapping per-bar P&L series (each 1-bar return enters exactly
    # once), so the Sharpe annualisation is unbiased — unlike ``pos × h-bar forward
    # return``, which overlaps ``hold−1`` bars and inflates Sharpe by ~``√hold``.
    # ``hold`` defaults to the forecast horizon; any residual P&L autocorrelation is
    # then genuine position persistence, not a measurement artifact.
    hold = cfg.holding_period or cfg.target_horizon
    book = pos.rolling(hold, min_periods=1).mean() if hold > 1 else pos
    pnl = book * forward_returns(close, horizon=1)

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
    is_m = _slice_metrics(pnl[is_mask], book[is_mask], sig[is_mask], close[is_mask], cfg)
    oos_m = _slice_metrics(pnl[oos_mask], book[oos_mask], sig[oos_mask], close[oos_mask], cfg)

    return {
        "is_ic": is_m.get("ic_mean"), "oos_ic": oos_m.get("ic_mean"),
        "is_sharpe": is_m.get("sharpe"), "oos_sharpe": oos_m.get("sharpe"),
        "is_ann_return": is_m.get("ann_return"), "oos_ann_return": oos_m.get("ann_return"),
        "oos_max_drawdown": oos_m.get("max_drawdown"), "oos_hit_rate": oos_m.get("hit_rate"),
        "oos_sharpe_std": oos_m.get("sharpe_std"),  # per_underlying aggregation only
    }
