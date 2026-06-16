"""Stocks experiencing a surge in hidden volume often indicate increased interest from informed traders, which may predispose the stock to a price movement. This factor aims to capture the pressure exerted by hidden trades, which may lead to future price changes, leveraging the insights from microstructure theory that suggest hidden liquidity can signal impending volatility."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, stddev

@register_factor
class HiddenVolumePressureIndicator(BaseFactor):
    factor_id = "hidden_volume_pressure_indicator"
    name = "Hidden Volume Pressure Indicator"
    category = "microstructure"
    inputs = ["hidden", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        hidden_volume = data["hidden"].fillna(0.0)
        total_volume = data["volume"].fillna(0.0)

        # Calculate rolling sums and means
        hidden_sum = ts_sum(hidden_volume, 60)
        total_sum = ts_sum(total_volume, 60)
        hidden_mean = ts_mean(hidden_volume, 60)
        hidden_stddev = stddev(hidden_volume, 60)

        # Calculate the ratio of hidden volume to total volume
        volume_ratio = hidden_sum / total_sum.replace(0, np.nan)

        # Normalize by the standard deviation of hidden volume
        pressure_indicator = volume_ratio / hidden_stddev.replace(0, np.nan)

        return pressure_indicator