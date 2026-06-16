"""Informed traders often use hidden orders to accumulate positions without causing significant market impact. By analyzing the ratio of hidden volume pressure to overall trading volume, we can identify stocks with strong hidden accumulation or distribution, suggesting potential price movements. This factor leverages the premise that hidden volume can be a leading indicator of price trends, particularly in illiquid stocks."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, abs_, delay


@register_factor
class DynamicHiddenVolumePressureRatio(BaseFactor):
    factor_id = "dynamic_hidden_volume_pressure_ratio"
    name = "Dynamic Hidden Volume Pressure Ratio"
    category = "microstructure"
    inputs = ["hidden", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        hidden_volume = data["hidden"].fillna(0.0)
        total_volume = data["volume"].fillna(0.0)

        # Calculate dynamic hidden volume pressure (net hidden volume)
        hidden_pressure = hidden_volume.where(hidden_volume > 0, 0.0)
        hidden_pressure_sum = ts_sum(hidden_pressure, 5)

        # Calculate total volume over the same period
        total_volume_sum = ts_sum(total_volume, 5)

        # Avoid division by zero by replacing zeros in total volume with NaN
        total_volume_sum = total_volume_sum.replace(0, np.nan)

        # Calculate the ratio of hidden volume pressure to total volume
        ratio = hidden_pressure_sum / total_volume_sum

        return ratio