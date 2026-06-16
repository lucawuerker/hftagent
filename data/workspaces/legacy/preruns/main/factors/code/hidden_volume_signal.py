"""Hidden volume can indicate the intent of informed traders who prefer not to reveal their orders to the market. By analyzing the relationship between hidden volume and price movement, we can exploit the potential price impact of undisclosed trades, capturing alpha from these informed signals."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import correlation, delta, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class HiddenVolumeSignal(BaseFactor):
    factor_id = "hidden_volume_signal"
    name = "Hidden Volume Signal"
    category = "microstructure"
    inputs = ["hidden", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        hidden_volume = data["hidden"].fillna(0.0)
        close = data["close"]
        price_change = delta(close, 1).fillna(0.0)
        hidden_volume_sum = ts_sum(hidden_volume, 5).replace(0, np.nan)
        correlation_signal = correlation(hidden_volume_sum, price_change, 5)
        return correlation_signal