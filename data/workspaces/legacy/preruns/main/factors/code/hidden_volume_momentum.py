"""Hidden volume, or the volume of trades that occurs away from the visible market, can signal future price movements. As suggested by literature on inventory management and partially observed demand, a buildup of hidden volume may indicate an accumulation of buying or selling pressure that is not yet reflected in the market price, thus providing a momentum signal."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, delta, rank


@register_factor
class HiddenVolumeMomentum(BaseFactor):
    factor_id = "hidden_volume_momentum"
    name = "Hidden Volume Momentum"
    category = "momentum"
    inputs = ["hidden", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        hidden_volume = data["hidden"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        # Calculate the momentum of hidden volume over a 5-bar window
        hidden_volume_momentum = delta(hidden_volume, 1)  # Change in hidden volume
        historical_avg_hidden_volume = ts_mean(hidden_volume, 5)  # 5-bar average of hidden volume

        # Normalize the momentum by historical average volume
        momentum_signal = hidden_volume_momentum / (historical_avg_hidden_volume.replace(0, np.nan))

        # Rank the momentum signal across the stocks
        ranked_signal = rank(momentum_signal)

        return ranked_signal