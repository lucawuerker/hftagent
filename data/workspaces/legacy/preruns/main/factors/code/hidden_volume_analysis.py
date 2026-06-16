"""Hidden orders can significantly impact price movements without appearing in standard volume metrics. By analyzing the hidden volume in conjunction with order flow, we can identify potential price reversals or continuations, leveraging the information asymmetry that exists in the market. This aligns with literature suggesting that hidden volume often precedes significant price changes."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, abs_, rank

@register_factor
class HiddenVolumeAnalysis(BaseFactor):
    factor_id = "hidden_volume_analysis"
    name = "Hidden Volume Analysis"
    category = "microstructure"
    description = "This signal computes the ratio of hidden volume to total volume across a specified time frame, identifying stocks with high levels of hidden trading activity relative to their overall trading volume. It then normalizes this value to provide a cross-sectional view of stocks."
    inputs = ["hidden", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        hidden_volume = data["hidden"].fillna(0.0)
        total_volume = data["volume"].fillna(0.0)
        total_volume = total_volume.replace(0, np.nan)  # Avoid division by zero

        hidden_volume_sum = ts_sum(hidden_volume, 5)
        total_volume_sum = ts_sum(total_volume, 5)

        ratio = hidden_volume_sum / total_volume_sum
        ratio = ratio.where(~total_volume_sum.isna(), 0.0)  # Set to 0 where total volume is NaN

        return rank(ratio)