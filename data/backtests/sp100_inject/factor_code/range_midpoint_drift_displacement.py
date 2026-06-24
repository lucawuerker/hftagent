"""When price closes far from the intrabar midpoint but the path itself was not particularly volatile, it often reflects temporary dealer inventory pressure or short-lived liquidity imbalance rather than durable information. The idea is to fade the displacement when the close has overshot the bar’s internal center after conditioning on the bar’s own range and volume, because prices that travel inefficiently within a bar tend to partially revert once the flow shock passes. This complements momentum-style seeds by focusing on the distance between where price ended and where the bar spent its time, rather than on direction alone."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank, ts_mean, ts_rank, adv
from quant_fund_agent.factors.registry import register_factor


@register_factor
class RangeMidpointDriftDisplacement(BaseFactor):
    """Cross-sectional mean-reversion signal from close displacement vs. intrabar midpoint."""

    factor_id = "range_midpoint_drift_displacement"
    name = "Range Midpoint Drift Displacement"
    category = "mean_reversion"
    description = (
        "Computes the cross-sectional z-scored signed distance of close from "
        "the bar midpoint, scaled by realized range and volume normalization."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        bar_mid = (high + low) * 0.5
        bar_range = (high - low).abs()
        safe_range = bar_range.replace(0.0, np.nan)

        # Signed displacement of the close from the intrabar midpoint, normalized by range.
        midpoint_displacement = (close - bar_mid) / safe_range

        # Penalize bars where volume is elevated relative to its recent baseline.
        vol_baseline = ts_mean(volume.fillna(0.0), 20)
        vol_ratio = volume / vol_baseline.replace(0.0, np.nan)
        volume_penalty = ts_rank(vol_ratio.fillna(0.0), 20)

        # Add a mild persistence filter so isolated one-bar spikes are downweighted.
        close_momentum = ts_rank(close - open_, 10)

        raw = -midpoint_displacement * (1.0 - 0.5 * volume_penalty) * (1.0 - 0.25 * close_momentum)
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Cross-sectional normalization to keep the signal comparable across the universe.
        out = rank(raw)
        return out.reindex(index=close.index, columns=close.columns)
