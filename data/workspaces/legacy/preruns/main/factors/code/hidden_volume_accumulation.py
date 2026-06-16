"""Hidden volume can indicate the presence of informed trading, where large trades are executed discreetly to avoid impacting the market. By observing the accumulation of hidden volume over short intervals, we can infer potential future price movements, as it reflects the strength of demand or supply pressure that isn't immediately visible in the order book. This signal computes the cumulative hidden volume over predefined short time frames, signaling when there is significant accumulation that may precede price movements. A higher accumulation of hidden volume might indicate stronger buying or selling interest, suggesting a potential price direction."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class HiddenVolumeAccumulation(BaseFactor):
    factor_id = "hidden_volume_accumulation"
    name = "Hidden Volume Accumulation"
    category = "microstructure"
    inputs = ["hidden"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        hidden_volume = data["hidden"].fillna(0.0)
        cumulative_hidden_volume = ts_sum(hidden_volume, 5)
        return cumulative_hidden_volume