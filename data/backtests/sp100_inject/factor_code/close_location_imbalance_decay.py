"""Compute the close location value ((close - low) / (high - low)) and weight it by a standardized volume surprise versus a rolling benchmark. Cross-sectionally, this produces a directional pressure score that is strongest for names closing near the bar high on abnormally high volume and weakest for names closing near the bar low on abnormally high volume."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank, ts_mean, stddev
from quant_fund_agent.factors.registry import register_factor


@register_factor
class CloseLocationImbalanceDecay(BaseFactor):
    """Close-location pressure weighted by volume surprise and smoothed by decay."""

    factor_id = "close_location_imbalance_decay"
    name = "Close-Location Imbalance Decay"
    category = "microstructure"
    inputs = ["close", "high", "low", "volume"]
    window_length = 20

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"].fillna(0.0)

        price_range = (high - low).replace(0, np.nan)
        close_location = (close - low) / price_range
        close_location = close_location.fillna(0.5).clip(0.0, 1.0)

        vol_mean = ts_mean(volume, 20)
        vol_std = stddev(volume, 20)
        vol_surprise = (volume - vol_mean) / (vol_std.replace(0, np.nan))
        vol_surprise = vol_surprise.fillna(0.0)

        raw = (close_location - 0.5) * vol_surprise
        signal = rank(raw)

        return signal
