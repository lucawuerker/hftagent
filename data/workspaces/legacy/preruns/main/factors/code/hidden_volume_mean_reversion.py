"""Hidden liquidity often signals large institutional trading activity that can lead to mean reverting price movements once the market absorbs this liquidity. When hidden volume is disproportionately high relative to recent averages, it can indicate that prices may revert after an initial reaction to such trades."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, sign

@register_factor
class HiddenVolumeMeanReversion(BaseFactor):
    factor_id = "hidden_volume_mean_reversion"
    name = "Hidden Volume Mean Reversion"
    category = "mean_reversion"
    inputs = ["hidden", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        hidden_volume = data["hidden"].fillna(0.0)
        total_volume = data["volume"].fillna(0.0)
        avg_hidden_volume = ts_mean(hidden_volume, 60)
        avg_total_volume = ts_mean(total_volume, 60)
        hidden_volume_ratio = hidden_volume / total_volume.replace(0, np.nan)
        mean_reversion_signal = hidden_volume_ratio - (avg_hidden_volume / avg_total_volume)
        return mean_reversion_signal