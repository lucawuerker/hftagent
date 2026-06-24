"""The Hawkes process suggests that past price movements and trades can increase the likelihood of future price movements, creating self-excitement in the market. By measuring the ratio of past intensities of price movements to current intensities, we can capture moments of heightened activity that may lead to further price momentum."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, delta, ts_mean, sign, abs_
from quant_fund_agent.factors.registry import register_factor


@register_factor
class HawkesExcitementRatio(BaseFactor):
    factor_id = "hawkes_excitement_ratio"
    name = "Hawkes Excitement Ratio"
    category = "momentum"
    description = "This factor computes the ratio of the current Hawkes intensity to the average intensity over a specified past period, capturing the degree of excitement in price movements."
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        intensity_current = delta(close, 1)  # Current intensity as the change in close price
        intensity_past_sum = ts_sum(abs_(delta(close, 1)), 5)  # Sum of past intensities over 5 bars
        intensity_average = ts_mean(intensity_past_sum, 5)  # Average intensity over the past 5 bars
        excitement_ratio = intensity_current / intensity_average.where(intensity_average != 0, np.nan)  # Avoid division by zero
        return excitement_ratio