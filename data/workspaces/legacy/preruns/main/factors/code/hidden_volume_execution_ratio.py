"""Research indicates that hidden orders can reduce the market impact of large trades, allowing informed traders to execute without revealing their intentions (Kashyap, 2014). By analyzing the ratio of hidden volume to total execution volume, we can identify stocks where large trades are likely being absorbed quietly, suggesting less price impact and potentially higher future returns."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, rank

@register_factor
class HiddenVolumeExecutionRatio(BaseFactor):
    factor_id = "hidden_volume_execution_ratio"
    name = "Hidden Volume to Total Execution Ratio"
    category = "microstructure"
    inputs = ["hidden", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        hidden_volume = data["hidden"].fillna(0.0)
        total_volume = data["volume"].fillna(0.0)

        # Calculate total execution volume over the last 5 bars
        total_execution_volume = ts_sum(total_volume, 5)

        # Calculate the ratio of hidden volume to total execution volume
        hidden_volume_ratio = hidden_volume / total_execution_volume.replace(0, np.nan)

        # Rank the hidden volume ratio
        ranked_ratio = rank(hidden_volume_ratio)

        return ranked_ratio