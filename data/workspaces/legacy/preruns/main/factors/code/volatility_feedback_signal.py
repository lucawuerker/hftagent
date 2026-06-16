"""Research indicates that market volatility tends to exhibit feedback effects, where recent increases in volatility can lead to further increases in volatility due to heightened uncertainty among investors (Yue et al., 2023). This suggests that stocks experiencing a spike in volatility may continue to do so, creating trading opportunities for short-term strategies that capitalize on this persistence."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (  
    delta, ts_sum, stddev, ts_rank, sign,  
)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class VolatilityFeedbackSignal(BaseFactor):
    factor_id = "volatility_feedback_signal"
    name = "Volatility Feedback Signal"
    category = "volatility"
    description = "This signal computes the normalized change in realized volatility over the past few 10-second bars, adjusted for the implied volatility levels."
    inputs = ["close", "volume", "effSpread"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        eff_spread = data["effSpread"].fillna(0.0)

        # Calculate the realized volatility over the past 5 bars
        realized_volatility = stddev(close, 5)

        # Calculate the change in realized volatility
        change_in_volatility = delta(realized_volatility, 1)

        # Normalize the change in volatility by the average volume
        avg_volume = ts_sum(volume, 5) / 5
        normalized_change = change_in_volatility / avg_volume.replace(0, np.nan)

        # Calculate the volatility momentum signal
        signal = ts_rank(normalized_change, 60)

        return signal