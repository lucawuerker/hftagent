"""This factor leverages the concept that price movements accompanied by high volume tend to indicate stronger momentum. The cumulative volume weighted price momentum can capture the persistence of trends in price movements, suggesting that stocks with higher volume-weighted price increases are likely to continue their upward trajectory, while those with decreases may continue downward. This aligns with behavioral finance theories suggesting that high volume indicates higher investor conviction."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, vwap


@register_factor
class CumulativeVolumeWeightedPriceMomentum(BaseFactor):
    factor_id = "cumulative_volume_weighted_price_momentum"
    name = "Cumulative Volume Weighted Price Momentum"
    category = "momentum"
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        # Calculate the volume weighted price
        vwp = vwap(data)

        # Calculate cumulative volume weighted price momentum
        cum_vwp = ts_sum(vwp, 5)
        cum_volume = ts_sum(volume, 5)

        # Calculate the momentum as cumulative volume weighted price changes
        momentum = cum_vwp / cum_volume

        # Replace NaN values with 0.0 for safety
        momentum = momentum.fillna(0.0)

        return momentum