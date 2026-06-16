"""The liquidity order flow ratio captures the relationship between incoming order flow and the liquidity available at the best bid and ask prices. By analyzing the balance of liquidity in relation to order flow, traders can identify potential price movements as imbalances indicate underlying market sentiment and pressure. Such dynamics are essential for predicting short-term price changes, as noted in order book research."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import (rank, delta, ts_sum, ts_mean, ts_max, ts_min, ts_argmax, ts_argmin, ts_rank, returns, vwap)

@register_factor
class LiquidityOrderFlowRatio(BaseFactor):
    factor_id = "liquidity_order_flow_ratio"
    name = "Liquidity Order Flow Ratio"
    category = "microstructure"
    inputs = ["orderFlow", "depth"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        depth = data["depth"].replace(0, np.nan)
        ratio = order_flow / depth
        return ratio.where(depth.notna(), 0.0)