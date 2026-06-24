"""As highlighted in the paper by Deschatre and Warin, the correlation between electricity prices for different maturities tends to decay as the time between them increases, which is known as the Samuelson correlation effect. This suggests that price movements for closely related maturities are likely to influence each other more than those further apart, creating potential arbitrage opportunities as traders react to price shocks in a non-uniform manner. By identifying and trading on the relative strength of this correlation decay, we can capture mean-reverting behavior in prices."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import correlation, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntradayPriceCorrelationDecay(BaseFactor):
    factor_id = "intraday_price_correlation_decay"
    name = "Intraday Price Correlation Decay"
    category = "statistical_arbitrage"
    description = "This factor computes the correlation between the prices of different maturities and normalizes it based on their time-to-maturity distance."
    inputs = ["close"]
    window_length = 60

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        n = close.shape[1]  # number of tickers
        correlation_matrix = pd.DataFrame(index=close.index, columns=close.columns)

        for i in range(n):
            for j in range(i + 1, n):
                corr = correlation(close.iloc[:, i], close.iloc[:, j], 60)
                correlation_matrix.iloc[:, i] = correlation_matrix.iloc[:, i].fillna(corr)
                correlation_matrix.iloc[:, j] = correlation_matrix.iloc[:, j].fillna(corr)

        decay_factor = ts_sum(correlation_matrix, 60)
        return decay_factor