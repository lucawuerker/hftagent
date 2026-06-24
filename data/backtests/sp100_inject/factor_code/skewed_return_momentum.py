"""Stocks often exhibit a momentum effect where past performance influences future returns. This factor leverages the skewness of returns, as suggested by the skew-elliptical distributions discussed in the literature, which capture asymmetric market behavior, particularly in reaction to news events or earnings announcements. Stocks with positive skewness in returns may continue to trend upwards, while those with negative skewness can indicate potential downturns. This signal computes the momentum of stock returns while adjusting for skewness, using the ratio of the current return to the historical average return, normalized by the skewness of the return distribution. The factor is designed to identify stocks likely to continue their recent performance trends based on skewness dynamics."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (rank, returns, ts_mean, ts_rank, stddev)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class SkewedReturnMomentum(BaseFactor):
    factor_id = "skewed_return_momentum"
    name = "Skewed Return Momentum"
    category = "momentum"
    description = "Momentum of stock returns adjusted for skewness."
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        returns_df = returns(data)
        avg_return = ts_mean(returns_df, 20)
        skewness = (returns_df - avg_return).apply(lambda x: np.sign(x) * (x ** 3).mean(), axis=0) / stddev(returns_df, 20)**3
        momentum_signal = returns_df / avg_return
        adjusted_signal = momentum_signal / (1 + skewness)
        return ts_rank(adjusted_signal, 20)