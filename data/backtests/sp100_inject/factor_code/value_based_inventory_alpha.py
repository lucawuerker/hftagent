"""The management of inventory can significantly impact a firm's value, as tied-up capital in excess inventory incurs opportunity costs. By analyzing the price movements relative to volume and the implied volatility associated with inventory levels, we can identify mispricings that arise from inefficient inventory management strategies, which can result in profitable trading opportunities. This is grounded in the insights from Michalski's work on value-based inventory management."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, delta, adv, vwap

@register_factor
class ValueBasedInventoryAlpha(BaseFactor):
    factor_id = "value_based_inventory_alpha"
    name = "Value-Based Inventory Alpha"
    category = "other"
    description = ("This signal computes the ratio of price change to the change in trading volume, "
                   "adjusted for the average price over a specified period, indicating the sensitivity "
                   "of a stock to its inventory management practices. A higher ratio signals potential "
                   "mispricing due to poor inventory handling, suggesting a mean-reversion opportunity.")
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        adv5 = adv(volume, 5)
        price_change = delta(close, 1)
        volume_change = delta(volume, 1)
        adjusted_price = vwap(data)  # Using VWAP as the average price
        ratio = price_change / volume_change.replace(0, np.nan)  # Avoid division by zero
        signal = ratio * adjusted_price
        return signal.where(adv5 > volume, 0.0)  # Fallback to 0.0 when volume is below ADV
