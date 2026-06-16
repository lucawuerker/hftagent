"""Hidden volume can indicate market sentiment and potential price movements, especially during periods of high volatility. By dynamically assessing the hidden volume pressure over a short time frame, traders can gauge whether the market is leaning towards buying or selling pressure, thus anticipating potential price changes. This approach aligns with the idea that latent trading activity often precedes significant price movements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean

@register_factor
class DynamicHiddenVolumePressure(BaseFactor):
    factor_id = "dynamic_hidden_volume_pressure"
    name = "Dynamic Hidden Volume Pressure"
    category = "microstructure"
    description = "This factor computes the ratio of hidden volume to total volume, adjusted over a rolling window, to capture shifts in buying or selling pressure. A rising ratio suggests increasing hidden buying pressure, while a falling ratio indicates increasing selling pressure."
    inputs = ["hidden", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        hidden_volume = data["hidden"].fillna(0.0)
        total_volume = data["volume"].replace(0, np.nan)
        total_volume = total_volume.fillna(0.0)

        hidden_volume_pressure = hidden_volume / total_volume
        hidden_volume_pressure = hidden_volume_pressure.replace(np.inf, np.nan)

        rolling_hidden_pressure = ts_mean(hidden_volume_pressure, 10)
        return rolling_hidden_pressure