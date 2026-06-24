"""The discrete-time Hawkes process can effectively capture self-exciting behavior in asset price movements, where past price changes influence future returns. By leveraging this property, we can identify stocks that exhibit momentum due to persistent buying or selling pressure, thus generating potential alpha. As shown in the literature, understanding the clustering effects of trading activity can lead to profitable trading strategies."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (delta, ts_sum, sign, adv)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class DiscreteHawkesMomentumSignal(BaseFactor):
    factor_id = "discrete_hawkes_momentum_signal"
    name = "Discrete Hawkes Momentum Signal"
    category = "momentum"
    inputs = ["close", "volume"]
    
    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        
        # Calculate price changes
        price_change = delta(close, 1)
        
        # Calculate the intensity of the Hawkes process
        intensity = sign(price_change) * volume
        
        # Cumulative intensity over the last 5 bars
        cumulative_intensity = ts_sum(intensity, 5)
        
        # Normalize the cumulative intensity
        normalized_signal = cumulative_intensity / adv(volume, 20).replace(0, np.nan)
        
        return normalized_signal