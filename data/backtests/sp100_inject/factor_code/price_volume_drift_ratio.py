"""This factor leverages the relationship between price movement and trading volume to identify potential continuation or reversal patterns. When the price increases significantly with high volume, it suggests strong bullish sentiment, whereas a price drop with high volume indicates bearish sentiment. These dynamics are rooted in behavioral finance, which posits that traders often overreact to volume spikes, leading to momentum effects."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (delta, adv, ts_sum)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class PriceVolumeDriftRatio(BaseFactor):
    factor_id = "price_volume_drift_ratio"
    name = "Price-Volume Drift Ratio"
    category = "momentum"
    description = ("The signal computes the ratio of the price change to the volume change over a specified period, capturing the drift in price relative to the trading activity. It identifies instances where price changes are disproportionately supported by volume, signaling potential future price movements.")
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        price_change = delta(close, 5)
        volume_change = delta(volume, 5)
        volume_change = volume_change.replace(0, np.nan)  # Avoid division by zero
        drift_ratio = price_change / volume_change
        return drift_ratio