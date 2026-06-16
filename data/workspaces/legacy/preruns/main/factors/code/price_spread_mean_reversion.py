"""In intraday trading, prices tend to revert to their mean after experiencing significant deviations. This mean-reversion effect can be driven by traders correcting mispricings or reacting to new information. By analyzing the spread between a stock's price and its moving average over a short time frame, we can identify potential reversal points and capitalize on these corrections.

This factor calculates the difference between the current price and its 10-second moving average, indicating how far the price has deviated from its recent average. A larger difference suggests a higher likelihood of mean reversion.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_mean, delta


@register_factor
class PriceSpreadMeanReversion(BaseFactor):
    factor_id = "price_spread_mean_reversion"
    name = "Price Spread Mean Reversion"
    category = "mean_reversion"
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        moving_avg = ts_mean(close, 10)
        spread = close - moving_avg
        return spread.where(spread.notna(), 0.0)