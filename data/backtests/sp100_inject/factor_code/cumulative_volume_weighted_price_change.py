"""This factor leverages the cumulative price change weighted by volume to identify trends driven by significant trading activity. The rationale is that price movements accompanied by high volume tend to signal stronger momentum, as they reflect genuine interest from market participants, thus potentially leading to sustained price trends."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, delta, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class CumulativeVolumeWeightedPriceChange(BaseFactor):
    factor_id = "cumulative_volume_weighted_price_change"
    name = "Cumulative Volume-Weighted Price Change"
    category = "momentum"
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        # Calculate cumulative price change weighted by volume
        price_change = delta(close, 1)  # Daily price change
        weighted_price_change = price_change * volume
        cumulative_weighted_price_change = ts_sum(weighted_price_change, 5)  # Cumulative over 5 bars

        # Normalize by cumulative volume
        cumulative_volume = ts_sum(volume, 5)
        result = cumulative_weighted_price_change / cumulative_volume.where(cumulative_volume != 0, np.nan)

        return result