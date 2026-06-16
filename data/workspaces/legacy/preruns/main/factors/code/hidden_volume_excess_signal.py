"""Hidden volume, which represents trades not visible in the order book, can provide insight into underlying market dynamics. An increase in hidden volume relative to visible volume may indicate that informed traders are entering their positions, potentially leading to future price movements. This is particularly relevant around key news events or earnings reports, where hidden order flow can precede significant price changes."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class HiddenVolumeExcessSignal(BaseFactor):
    factor_id = "hidden_volume_excess_signal"
    name = "Hidden Volume Excess Signal"
    category = "microstructure"
    description = ("This signal computes the ratio of hidden volume to total volume across a rolling window, allowing for the identification of periods where hidden volume disproportionately influences trading activity. A higher ratio indicates greater hidden volume relative to visible volume, suggesting potential market impact.")
    inputs = ["hidden", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        hidden_volume = data["hidden"].fillna(0.0)
        total_volume = data["volume"].fillna(0.0)
        hidden_volume_ratio = hidden_volume / total_volume.replace(0, np.nan)
        hidden_volume_ratio = hidden_volume_ratio.where(~hidden_volume_ratio.isna(), 0.0)
        return ts_mean(hidden_volume_ratio, 60)