"""The dynamics of hidden volume can provide insights into the underlying order flow pressures in the market. High ratios of hidden volume to visible volume may indicate traders are accumulating positions discreetly, often preceding price movements, which aligns with concepts discussed in market microstructure literature. A deeper understanding of hidden volume can thus exploit imbalances in supply and demand.

This factor computes the ratio of hidden traded volume to total traded volume over a specified period, providing a measure of the quiet accumulation of positions in the market relative to overall activity."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum


@register_factor
class HiddenVolumePressureRatio(BaseFactor):
    factor_id = "hidden_volume_pressure_ratio"
    name = "Hidden Volume Pressure Ratio"
    category = "microstructure"
    inputs = ["hidden", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        hidden_volume = data["hidden"].fillna(0.0)
        total_volume = data["volume"].fillna(0.0)

        # Calculate the total volume over the last 5 bars
        total_volume_sum = ts_sum(total_volume, 5)

        # Replace zeros in total volume to avoid division by zero
        total_volume_sum = total_volume_sum.replace(0, np.nan)

        # Calculate the ratio of hidden volume to total volume
        ratio = hidden_volume / total_volume_sum

        return ratio