"""Stocks that exhibit high relative volume alongside positive price momentum tend to attract more attention from both retail and institutional investors, leading to further price increases. This behavior can be linked to the 'bandwagon effect' in behavioral finance, where investors are drawn to stocks that are experiencing upward price movements coupled with increased trading activity, often leading to a self-reinforcing cycle of buying."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import adv, delta, ts_mean, ts_rank


@register_factor
class RelativeVolumePriceMomentum(BaseFactor):
    factor_id = "relative_volume_price_momentum"
    name = "Relative Volume Price Momentum"
    category = "momentum"
    description = ("This signal calculates the ratio of current volume to a moving average of past volumes (e.g., 20 days) for each stock, and combines it with the recent price momentum (e.g., a 5-day price change) to identify stocks that are both gaining in price and experiencing higher-than-average trading volume.")
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        adv20 = adv(volume, 20)
        price_momentum = delta(close, 5)
        relative_volume = volume / adv20.fillna(1.0)  # Avoid division by zero
        signal = ts_rank(relative_volume * price_momentum, 60)
        return signal.fillna(0.0)