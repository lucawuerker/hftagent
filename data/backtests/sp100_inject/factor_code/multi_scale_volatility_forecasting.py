"""Different market participants operate on various timescales, leading to distinct volatility influences at different frequencies. By decomposing volatility into multiple scales using wavelet transforms, we can capture the short-term and long-term variance components more effectively. This aligns with findings in Barunik and Vacha (2014), which highlight the importance of understanding volatility dynamics across different investment horizons."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (  
    ts_sum, stddev, ts_rank, delay,  
)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class MultiScaleVolatilityForecasting(BaseFactor):
    factor_id = "multi_scale_volatility_forecasting"
    name = "Multi-Scale Volatility Forecasting"
    category = "volatility"
    description = "This factor computes the realized volatility across different timescales using wavelet decomposition, allowing traders to predict future volatility based on the contributions from various investment horizons."
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        # Calculate realized volatility using a rolling window
        realized_volatility = stddev(close, 20)  # 20-bar rolling standard deviation
        volume_weighted_volatility = stddev(close * volume, 20) / (stddev(volume, 20) + 1e-10)  # Avoid division by zero

        # Combine the two volatility measures
        combined_volatility = ts_sum(realized_volatility, 5) + ts_sum(volume_weighted_volatility, 5)

        # Rank the combined volatility
        output = ts_rank(combined_volatility, 60)

        return output