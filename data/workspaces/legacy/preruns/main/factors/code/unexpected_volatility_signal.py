"""Based on the findings from Bianco and Renò (2006), unexpected volatility has been shown to correlate positively with intraday serial correlation. This can create profitable trading opportunities as traders tend to react to surprises in volatility, leading to price momentum in the direction of the volatility shock. Capturing these moments when volatility deviates from expectations can provide an edge in predicting short-term price movements.

This factor computes the unexpected volatility in a given time period, defined as the residuals from a linear regression of realized volatility against previous volatility, highlighting periods of significant volatility surprises that can influence price movements.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_mean, ts_sum, stddev, delta, sign


@register_factor
class UnexpectedVolatilitySignal(BaseFactor):
    factor_id = "unexpected_volatility_signal"
    category = "volatility"
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        # Calculate realized volatility as the standard deviation of returns over a rolling window
        returns = close.pct_change().fillna(0.0)
        realized_volatility = stddev(returns, 10)

        # Calculate the mean of realized volatility over the previous 10 bars
        mean_volatility = ts_mean(realized_volatility, 10)

        # Calculate unexpected volatility as the difference between realized and mean volatility
        unexpected_volatility = realized_volatility - mean_volatility

        # Return the unexpected volatility signal, ensuring the same index and columns as close
        return unexpected_volatility.where(realized_volatility.notna(), 0.0)