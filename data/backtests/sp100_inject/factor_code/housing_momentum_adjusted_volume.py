"""This factor leverages the concept that increased trading volume in housing markets often accompanies momentum in home prices, as suggested by Case and Shiller (1988). The idea is that heightened interest and speculative buying during price uptrends can drive prices higher, while low volume during downturns suggests a lack of conviction, leading to price declines."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, delta, ts_mean, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class HousingMomentumAdjustedVolume(BaseFactor):
    factor_id = "housing_momentum_adjusted_volume"
    name = "Housing Momentum Adjusted Volume"
    category = "momentum"
    inputs = ["close", "volume"]
    
    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        
        # Calculate the price momentum (current close - previous close)
        price_momentum = delta(close, 1)
        
        # Calculate the moving average of volume over the last 5 bars
        avg_volume = adv(volume, 5)
        
        # Calculate the ratio of current volume to the moving average of volume during positive price momentum
        volume_adjusted = volume.where(price_momentum > 0, 0) / avg_volume
        
        # Rank the volume adjusted values over the last 60 bars
        return ts_rank(volume_adjusted, 60)