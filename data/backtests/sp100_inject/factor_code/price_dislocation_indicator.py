"""Price dislocations occur when market prices deviate significantly from their intrinsic value, often due to overreaction from market participants. These deviations tend to revert to the mean over time as rational investors exploit the mispricing, creating opportunities for profit. This behavior is rooted in behavioral finance concepts, such as loss aversion and herd mentality."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_mean, delta, sign


@register_factor
class PriceDislocationIndicator(BaseFactor):
    factor_id = "price_dislocation_indicator"
    name = "Price Dislocation Indicator"
    category = "mean_reversion"
    description = "This signal computes the percentage difference between the current price and a moving average of the price over a specified period, identifying situations where the price is significantly above or below the average. A threshold can be set to determine when to signal a potential reversion."
    inputs = ["close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        moving_average = ts_mean(close, 20)
        price_difference = delta(close, 1)
        percentage_difference = price_difference / moving_average
        return percentage_difference.where(moving_average != 0, 0.0)