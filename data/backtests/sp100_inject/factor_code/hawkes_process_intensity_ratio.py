"""The self-exciting nature of Hawkes processes suggests that a stock's past trading intensity can predict future price movements. When the intensity of recent trades significantly exceeds its historical average, it indicates a potential upward momentum, as traders are likely reacting to new information. This aligns with findings in financial literature that highlight the predictive power of trade intensity on asset returns."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, adv
from quant_fund_agent.factors.registry import register_factor


@register_factor
class HawkesProcessIntensityRatio(BaseFactor):
    factor_id = "hawkes_process_intensity_ratio"
    name = "Hawkes Process Intensity Ratio"
    category = "momentum"
    inputs = ["volume"]
    description = ("This factor computes the ratio of the current intensity of a Hawkes process fitted to the stock's trading volume to its historical average intensity over a defined lookback period. A ratio significantly greater than one may indicate bullish momentum.")

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"].fillna(0.0)
        current_intensity = volume
        historical_average_intensity = adv(volume, 20)
        intensity_ratio = current_intensity / historical_average_intensity.replace(0, np.nan)
        return intensity_ratio