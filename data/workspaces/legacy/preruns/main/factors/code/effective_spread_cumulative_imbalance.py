"""This factor leverages the relationship between effective spread and cumulative order flow imbalance. A positive cumulative imbalance often precedes price moves, as observed in the literature detailing the dynamics of order flow and price impact. It can indicate the strength of buying or selling pressure, which is critical in predicting short-term price movements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import (
    ts_sum, ts_mean, ts_rank, 
    delay, sign, 
)

@register_factor
class EffectiveSpreadCumulativeImbalance(BaseFactor):
    factor_id = "effective_spread_cumulative_imbalance"
    name = "Effective Spread Cumulative Imbalance"
    category = "microstructure"
    inputs = ["effSpread", "effLobImb", "orderFlow"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        eff_spread = data["effSpread"].fillna(0.0)
        eff_lob_imb = data["effLobImb"].fillna(0.0)
        order_flow = data["orderFlow"].fillna(0.0)

        # Calculate cumulative order flow imbalance over the last 5 bars
        cumulative_imbalance = ts_sum(order_flow, 5)

        # Calculate the correlation between effective spread and cumulative imbalance
        correlation = ts_rank(cumulative_imbalance, 60) * ts_rank(eff_spread, 60)

        return correlation.where(eff_lob_imb != 0, 0.0)
