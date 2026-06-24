"""Market participants often exhibit collective behavior that leads to periodic changes in trading volume, as suggested by Sato (2006). By analyzing the variance in tick frequency over time, we can identify periods of increased or decreased market activity, which may signal shifts in market sentiment or impending price movements. This factor can capture the dynamics of market participation, providing insights into potential price trends."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, stddev

@register_factor
class TickFrequencyVariance(BaseFactor):
    factor_id = "tick_frequency_variance"
    name = "Tick Frequency Variance"
    category = "microstructure"
    description = "This factor computes the variance of tick frequency for each asset over a specified rolling window, allowing us to assess the stability or volatility of trading activity. It is calculated using the number of ticks over time, normalized by the average tick frequency."
    window_length = 60
    inputs = ["volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"].fillna(0.0)
        tick_frequency = volume / ts_mean(volume, 5)
        variance = stddev(tick_frequency, 60) ** 2
        return variance