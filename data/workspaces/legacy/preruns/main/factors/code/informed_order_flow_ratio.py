"""Informed traders often leave clues in the order flow, particularly when they increase their net buying or selling pressure. When the ratio of buy to sell order flow exceeds a certain threshold, it indicates the presence of informed trading, which can lead to price movements in the same direction. This aligns with the insights from Easley et al. (2012) regarding order flow toxicity and its impact on price discovery.

This factor computes the ratio of net buy order flow to total order flow over a specified period, indicating potential informed trading activity. A higher ratio suggests stronger informed buying, which may precede upward price movements.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, delta, abs_, sign


@register_factor
class InformedOrderFlowRatio(BaseFactor):
    factor_id = "informed_order_flow_ratio"
    category = "microstructure"
    inputs = ["orderFlow"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        total_order_flow = ts_sum(abs_(order_flow), 5)
        net_order_flow = ts_sum(order_flow, 5)
        ratio = net_order_flow / total_order_flow.replace(0, np.nan)
        return ratio