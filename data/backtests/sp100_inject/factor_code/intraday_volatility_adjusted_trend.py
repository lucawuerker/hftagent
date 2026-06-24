"""Higher volatility in intraday price movements often indicates increased trading interest and can signal momentum. By adjusting trends based on recent volatility, we can capture stronger price movements while avoiding false signals during low volatility periods. This aligns with behavioral finance principles, where traders tend to overreact to new information during volatile conditions."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_mean, ts_sum, stddev, delta, sign, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntradayVolatilityAdjustedTrend(BaseFactor):
    factor_id = "intraday_volatility_adjusted_trend"
    name = "Intraday Volatility Adjusted Trend"
    category = "momentum"
    description = ("This factor computes a volatility-adjusted trend by measuring the slope of the price movement over a defined lookback period, "
                   "scaled by the average intraday volatility. The result is a signal that indicates the strength and direction of momentum "
                   "relative to the volatility of price movements.")
    inputs = ["close", "volume"]
    window_length = 60

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        avg_vol = ts_mean(volume, 20)
        price_change = delta(close, 5)
        volatility = stddev(close, 20)
        trend = ts_sum(price_change, 5) / avg_vol
        adjusted_trend = trend / (volatility + 1e-10)  # Avoid division by zero
        return ts_rank(adjusted_trend, 60)