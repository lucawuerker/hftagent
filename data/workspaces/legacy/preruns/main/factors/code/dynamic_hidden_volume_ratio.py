"""The hidden volume can serve as a proxy for informed trading activity, as it often reflects the demand and supply dynamics that are not visible in the publicly quoted order book. By analyzing the ratio of hidden volume to total volume over rolling windows, we can capture shifts in trader sentiment and potential reversals in price trends, as informed traders may accumulate positions ahead of significant price movements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum

@register_factor
class DynamicHiddenVolumeRatio(BaseFactor):
    factor_id = "dynamic_hidden_volume_ratio"
    name = "Dynamic Hidden Volume Ratio"
    category = "microstructure"
    inputs = ["hidden", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        hidden_volume = data["hidden"].fillna(0.0)
        total_volume = data["volume"].replace(0, np.nan)

        # Calculate rolling sum of hidden volume and total volume over the last 5 bars
        hidden_sum = ts_sum(hidden_volume, 5)
        total_sum = ts_sum(total_volume, 5)

        # Calculate the ratio of hidden volume to total volume
        ratio = hidden_sum / total_sum

        # Replace NaN values with 0 for the final output
        return ratio.fillna(0.0)