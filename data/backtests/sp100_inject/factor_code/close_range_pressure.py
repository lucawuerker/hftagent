"""Intraday volatility is highly persistent and exhibits strong commonality across stocks, with the paper showing that the latest time-of-day volatility buckets carry disproportionate predictive power for both next-bar and next-day risk. A stock that repeatedly closes bars near the edge of its own high-low range, especially when accompanied by elevated turnover, is likely experiencing informed disagreement or inventory-driven pressure that tends to spill into the next interval. This factor is designed to capture that latent continuation in realized movement intensity rather than direction, complementing pure momentum or mean-reversion signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, rank, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class CloseRangePressure(BaseFactor):
    """Close-range pressure signal emphasizing persistent end-of-bar tension."""

    factor_id = "close_range_pressure"
    name = "Close Range Pressure"
    category = "volatility"
    description = (
        "Compute a signed range-pressure score from the close’s position within "
        "the bar range, scaled by the bar’s true range and volume relative to a "
        "rolling baseline. Higher values indicate persistent end-of-bar tension "
        "near the extremes with heavier activity, which is interpreted as higher "
        "near-term volatility continuation."
    )
    window_length = 20
    inputs = ["high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"].fillna(0.0)
        low = data["low"].fillna(0.0)
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        true_range = (high - low).abs()
        safe_range = true_range.replace(0, np.nan)

        # Signed pressure: positive when close is near the high, negative near the low.
        range_position = ((close - low) / safe_range) * 2.0 - 1.0
        range_position = range_position.where(np.isfinite(range_position), 0.0)

        # Normalize range pressure by current bar range relative to its recent baseline.
        tr_baseline = ts_mean(true_range, 20)
        range_intensity = (true_range / tr_baseline.replace(0, np.nan)).where(
            np.isfinite(true_range), 0.0
        )
        range_intensity = range_intensity.where(np.isfinite(range_intensity), 0.0)

        # Activity regime: heavier turnover relative to recent average amplifies the signal.
        vol_baseline = adv(volume, 20)
        vol_pressure = (volume / vol_baseline.replace(0, np.nan)).where(
            np.isfinite(volume), 0.0
        )
        vol_pressure = vol_pressure.where(np.isfinite(vol_pressure), 0.0)

        # Persistence proxy: bars repeatedly closing at the edge of their range matter more.
        edge_closeness = 1.0 - range_position.abs().clip(upper=1.0)
        edge_persistence = rank(edge_closeness)

        score = range_position * range_intensity * vol_pressure * (1.0 + edge_persistence)
        score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return score
