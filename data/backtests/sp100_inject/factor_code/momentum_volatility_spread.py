"""Higher momentum stocks often exhibit lower volatility, suggesting a market preference for stability in trending assets. This relationship can be exploited by taking long positions in high-momentum, low-volatility stocks while shorting low-momentum, high-volatility stocks, as investors tend to favor consistency in performance."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import (rank, delta, stddev, ts_mean, returns)

@register_factor
class MomentumVolatilitySpread(BaseFactor):
    factor_id = "momentum_volatility_spread"
    name = "Momentum Volatility Spread"
    category = "momentum"
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]

        # Calculate momentum as the rate of change in price over a defined period (e.g., 5 bars)
        momentum = delta(close, 5)

        # Calculate volatility as the standard deviation of returns over a defined period (e.g., 20 bars)
        returns_data = returns(data)
        volatility = stddev(returns_data, 20)

        # Calculate the spread between momentum and inverse of volatility
        inverse_volatility = 1 / (volatility.replace(0, np.nan))  # Avoid division by zero
        spread = momentum - inverse_volatility

        # Handle NaN values
        spread = spread.fillna(0.0)

        return spread