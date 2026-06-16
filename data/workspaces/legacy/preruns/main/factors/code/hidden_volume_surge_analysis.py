"""Hidden volume surges can indicate aggressive buying or selling pressure that may not be immediately visible in the order book. This can be a signal of impending price movements, as significant hidden orders may lead to price changes once they are revealed. Research indicates that hidden volume can capture market imbalances that precede price adjustments, making this a valuable predictor of future price movements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, delta, rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class HiddenVolumeSurgeAnalysis(BaseFactor):
    factor_id = "hidden_volume_surge_analysis"
    name = "Hidden Volume Surge Analysis"
    category = "microstructure"
    inputs = ["hidden"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        hidden_volume = data["hidden"].fillna(0.0)
        avg_hidden_volume = ts_mean(hidden_volume, 60)
        change_in_hidden_volume = delta(hidden_volume, 1)
        surge_signal = change_in_hidden_volume / avg_hidden_volume.replace(0, np.nan)
        return rank(surge_signal)