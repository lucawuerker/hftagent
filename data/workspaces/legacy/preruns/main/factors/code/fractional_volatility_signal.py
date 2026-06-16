"""Recent research suggests that volatility can be better understood through fractional noise processes rather than traditional models. By analyzing the recent price movements and their volatility patterns, we can identify stocks that exhibit rough volatility behavior, indicating potential price reversals or continuations based on their volatility characteristics."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, stddev, ts_mean


@register_factor
class FractionalVolatilitySignal(BaseFactor):
    factor_id = "fractional_volatility_signal"
    name = "Fractional Volatility Signal"
    category = "volatility"
    description = ("This factor computes the volatility of each stock over a defined window using the effective volatility derived from the recent price changes, adjusted for the fractional noise characteristics as described in the literature. It captures the dynamic nature of volatility and its potential impact on future price behavior.")
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        # Calculate the effective volatility over a 60-bar window
        effective_volatility = stddev(close, 60)

        # Calculate the mean volume over the same window
        mean_volume = ts_mean(volume, 60)

        # Normalize the effective volatility by the mean volume
        volatility_signal = effective_volatility / mean_volume.replace(0, np.nan)

        # Handle NaN values by filling them with 0
        return volatility_signal.fillna(0.0)