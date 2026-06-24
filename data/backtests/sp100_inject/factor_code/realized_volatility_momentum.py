"""High realized volatility can indicate increased investor uncertainty and market stress, often leading to price corrections. By capturing momentum in realized volatility, traders can identify assets that are in the midst of significant price movements, which may continue in the same direction, as suggested by the feedback effects documented in the literature."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_mean, stddev, delta, ts_rank


@register_factor
class RealizedVolatilityMomentum(BaseFactor):
    factor_id = "realized_volatility_momentum"
    name = "Realized Volatility Momentum"
    category = "momentum"
    description = ("This signal computes the momentum of realized volatility by calculating the rate of change of a realized volatility measure derived from high, low, and closing prices over a specified period. It identifies stocks that have exhibited increasing or decreasing volatility trends, potentially signaling a continuation in price movement.")
    inputs = ["high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"].fillna(0.0)
        low = data["low"].fillna(0.0)
        close = data["close"].fillna(0.0)

        # Calculate realized volatility as the standard deviation of returns
        realized_volatility = stddev(close, 20)  # 20-bar rolling standard deviation
        momentum = delta(realized_volatility, 5)  # 5-bar momentum of realized volatility

        # Calculate the rank of the momentum
        momentum_rank = ts_rank(momentum, 60)  # 60-bar rank of momentum

        return momentum_rank