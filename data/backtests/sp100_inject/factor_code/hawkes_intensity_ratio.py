"""The Hawkes process is a model for capturing the self-exciting nature of trades in financial markets, where past events influence future events. A ratio of the intensity of upward trades to downward trades can provide insights into market sentiment and momentum, potentially capturing the effects of trader behavior and order flow dynamics."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, delta, sign, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class HawkesIntensityRatio(BaseFactor):
    factor_id = "hawkes_intensity_ratio"
    name = "Hawkes Intensity Ratio"
    category = "microstructure"
    inputs = ["volume", "close"]
    
    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"].fillna(0.0)
        close = data["close"]
        
        # Calculate the delta of close prices to identify upward and downward movements
        close_delta = delta(close, 1)
        
        # Identify upward and downward trade volumes
        upward_volume = volume.where(close_delta > 0, 0)
        downward_volume = volume.where(close_delta < 0, 0)
        
        # Calculate the intensity of upward and downward trades over a rolling window
        upward_intensity = ts_sum(upward_volume, 5)
        downward_intensity = ts_sum(downward_volume, 5)
        
        # Calculate the Hawkes intensity ratio
        intensity_ratio = upward_intensity / downward_intensity.replace(0, np.nan)
        
        # Fill NaN values with 0.0 for the final output
        return intensity_ratio.fillna(0.0)