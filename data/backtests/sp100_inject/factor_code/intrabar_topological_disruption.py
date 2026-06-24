"""Intrabar Topological Disruption.

Construct a rolling persistence-based instability score from the intraday path using open/high/low/close over a lookback window, then cross-sectionally z-score it. Higher values indicate more topological disruption in the recent bar path relative to the asset's own history."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntrabarTopologicalDisruption(BaseFactor):
    """Rolling intrabar path instability from OHLC geometry."""

    factor_id = "intrabar_topological_disruption"
    name = "Intrabar Topological Disruption"
    category = "other"
    description = (
        "Construct a rolling persistence-based instability score from the intraday "
        "path using open/high/low/close over a lookback window, then cross-sectionally "
        "z-score it. Higher values indicate more topological disruption in the recent "
        "bar path relative to the asset's own history."
    )
    window_length = 30
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        eps = 1e-12

        bar_range = (high - low).abs()
        body = (close - open_).abs()
        upper_wick = (high - np.maximum(open_, close)).clip(lower=0.0)
        lower_wick = (np.minimum(open_, close) - low).clip(lower=0.0)

        # Geometric instability components: path elongation, wick asymmetry, and
        # frequent intrabar reversals between open/mid-close extremes.
        elongation = (body + upper_wick + lower_wick) / (bar_range + eps)
        wick_imbalance = (upper_wick - lower_wick).abs() / (bar_range + eps)

        prev_close = close.shift(1)
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_open = open_.shift(1)

        breakout_up = (high > prev_high) & (close < prev_close)
        breakout_down = (low < prev_low) & (close > prev_close)
        gap_reversal = ((open_ > prev_close) & (close < open_)) | ((open_ < prev_close) & (close > open_))
        intrabar_branch = breakout_up.astype(float) + breakout_down.astype(float) + gap_reversal.astype(float)

        # Recent instability state.
        local_state = 0.50 * elongation + 0.30 * wick_imbalance + 0.20 * intrabar_branch
        local_state = local_state.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Historical persistence proxy: when the local state departs from its own
        # trailing mean and variance, fragmentation is elevated.
        mu = local_state.rolling(self.window_length, min_periods=max(5, self.window_length // 3)).mean()
        sd = local_state.rolling(self.window_length, min_periods=max(5, self.window_length // 3)).std()
        instability = (local_state - mu) / (sd + eps)

        # Smooth and re-center cross-sectionally per bar.
        instability = instability.rolling(3, min_periods=1).mean()
        cross_mean = instability.mean(axis=1)
        cross_std = instability.std(axis=1).replace(0.0, np.nan)
        zscore = instability.sub(cross_mean, axis=0).div(cross_std, axis=0)

        return zscore.replace([np.inf, -np.inf], np.nan).fillna(0.0)
