"""The realized volatility of an asset is indicative of market uncertainty and can impact trading behavior. By comparing the intraday realized volatility of an asset to its recent historical volatility, traders may identify periods of heightened risk or potential price movement, allowing for more informed trading decisions. This approach aligns with the findings of the paper which emphasizes the importance of realized volatility in predicting future asset behavior."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_mean, stddev, ts_rank, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntradayRealizedVolatilityRatio(BaseFactor):
    factor_id = "intraday_realized_volatility_ratio"
    name = "Intraday Realized Volatility Ratio"
    category = "volatility"
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        # Calculate the intraday realized volatility as the standard deviation of returns
        returns = close.pct_change().fillna(0.0)
        realized_volatility = stddev(returns, 5)

        # Calculate the average realized volatility over the last 20 bars
        avg_realized_volatility = ts_mean(realized_volatility, 20)

        # Calculate the ratio of current realized volatility to average realized volatility
        volatility_ratio = realized_volatility / avg_realized_volatility.replace(0, np.nan)

        # Return the percentile rank of the volatility ratio over the last 60 bars
        return ts_rank(volatility_ratio, 60)