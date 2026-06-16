"""In high-frequency trading, aggressive hidden volume can indicate imminent price movements, as it suggests that informed traders are taking positions that could lead to significant price changes. This aligns with the findings in the paper 'Deep Limit Order Book Forecasting', which emphasize the predictive power of hidden volume in determining future price actions. This factor computes the ratio of aggressive hidden volume to total volume over a rolling window, capturing the intensity of informed trading activity. A higher value suggests increased buying or selling pressure from informed traders, potentially leading to future price movements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, abs_, delay
from quant_fund_agent.factors.registry import register_factor


@register_factor
class AggressiveHiddenVolumeSignal(BaseFactor):
    factor_id = "aggressive_hidden_volume_signal"
    name = "Aggressive Hidden Volume Signal"
    category = "microstructure"
    inputs = ["hidden", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        hidden_volume = data["hidden"].fillna(0.0)
        total_volume = data["volume"].replace(0, np.nan)

        # Calculate rolling sum of hidden volume and total volume over the last 5 bars
        hidden_volume_sum = ts_sum(hidden_volume, 5)
        total_volume_sum = ts_sum(total_volume, 5)

        # Calculate the ratio of aggressive hidden volume to total volume
        ratio = hidden_volume_sum / total_volume_sum

        # Handle NaN values in the ratio
        ratio = ratio.fillna(0.0)

        return ratio