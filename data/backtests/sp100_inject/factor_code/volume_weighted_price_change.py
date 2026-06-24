"""The volume weighted price change factor leverages the idea that price changes accompanied by higher trading volume tend to indicate stronger market sentiment, making it more likely that the price movement will persist. This reflects the concept that significant buying or selling pressure can lead to more pronounced price shifts, which can be profitable to exploit. As noted in market microstructure literature, higher volume can signal informed trading and support stronger price trends."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, delta, adv


@register_factor
class VolumeWeightedPriceChange(BaseFactor):
    factor_id = "volume_weighted_price_change"
    name = "Volume Weighted Price Change"
    category = "microstructure"
    description = "This factor computes the price change of a stock weighted by the volume traded during the same period, allowing for the assessment of how meaningful price movements are relative to the level of trading activity."
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        price_change = delta(close, 1)
        weighted_price_change = price_change * volume
        vwpc = weighted_price_change / adv(volume, 20)
        return vwpc