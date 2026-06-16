"""Hidden volume can indicate the presence of informed trading, as it often reflects traders' intention to enter or exit positions without revealing their full size to the market. By comparing hidden volume to visible volume, we can identify stocks that are likely to experience price movements due to underlying demand or supply pressure that is not immediately visible in the market."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum


@register_factor
class HiddenVolumeRatio(BaseFactor):
    factor_id = "hidden_volume_ratio"
    name = "Hidden Volume Ratio"
    category = "microstructure"
    inputs = ["hidden", "volume"]
    
    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        hidden_volume = data["hidden"].fillna(0.0)
        total_volume = data["volume"].replace(0, np.nan)
        
        # Calculate the ratio of hidden volume to total volume
        hidden_volume_ratio = hidden_volume / total_volume
        
        # Handle NaN values by filling them with 0.0
        hidden_volume_ratio = hidden_volume_ratio.fillna(0.0)
        
        return hidden_volume_ratio