"""This factor captures the relationship between hidden volume and the effective spread. A higher ratio may indicate that large orders are being executed at favorable prices, suggesting potential upward price pressure, while a lower ratio may suggest a lack of demand. This aligns with the concept that informed traders utilize hidden volume to mitigate market impact, as discussed in the literature on market microstructure."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum


@register_factor
class HiddenVolumeEffectiveSpreadRatio(BaseFactor):
    factor_id = "hidden_volume_effective_spread_ratio"
    name = "Hidden Volume to Effective Spread Ratio"
    category = "microstructure"
    inputs = ["hidden", "effSpread"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        hidden_volume = data["hidden"].fillna(0.0)
        effective_spread = data["effSpread"].replace(0, np.nan)

        # Calculate the sum of hidden volume over the last 5 bars
        hidden_volume_sum = ts_sum(hidden_volume, 5)
        # Calculate the sum of effective spread over the last 5 bars
        effective_spread_sum = ts_sum(effective_spread, 5)

        # Calculate the ratio, handling NaN values
        ratio = hidden_volume_sum / effective_spread_sum
        ratio = ratio.replace(np.inf, np.nan)  # replace inf with NaN

        return ratio