"""The presence of volatility clustering suggests that periods of high volatility are often followed by further high volatility, and vice versa. This phenomenon can be exploited by identifying stocks that experience sudden spikes in volatility, which may indicate that the market is reacting to new information or changing sentiment. Leveraging the insights from the paper on long-term memory and volatility clustering, we can capture potential price movements following such volatility spikes."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_mean, stddev, delta, sign
from quant_fund_agent.factors.registry import register_factor


@register_factor
class VolatilityClusteringSignal(BaseFactor):
    factor_id = "volatility_clustering_signal"
    name = "Volatility Clustering Signal"
    category = "volatility"
    description = "This factor computes the rolling standard deviation of price changes over a specified window, identifying stocks with significant deviations from their historical volatility levels."
    inputs = ["close"]
    window_length = 60

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        price_changes = delta(close, 1).replace(0, np.nan)
        rolling_std = stddev(price_changes, self.window_length)
        return rolling_std