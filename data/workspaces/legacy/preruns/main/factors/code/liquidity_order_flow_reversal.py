"""In periods of high liquidity, sudden changes in order flow can signal potential reversals in stock prices. A significant increase in buying or selling pressure may indicate that traders are attempting to close positions, suggesting an impending price correction. This phenomenon aligns with behavioral finance theories, where traders react to market conditions in a manner that can lead to overreactions and subsequent reversals.

This factor computes the ratio of order flow to liquidity (depth) and identifies abrupt changes, indicating potential reversals in price direction. It captures both the magnitude and direction of order flow relative to market liquidity, highlighting moments when liquidity conditions may be unsustainable.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, delta, sign


@register_factor
class LiquidityOrderFlowReversal(BaseFactor):
    factor_id = "liquidity_order_flow_reversal"
    name = "Liquidity Order Flow Reversal"
    category = "microstructure"
    inputs = ["orderFlow", "depth"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        depth = data["depth"].replace(0, np.nan)

        # Calculate the ratio of order flow to depth
        ratio = order_flow / depth

        # Calculate the change in the ratio
        delta_ratio = delta(ratio, 1)

        # Identify abrupt changes in order flow relative to liquidity
        signal = sign(delta_ratio) * ts_sum(abs(delta_ratio), 5)

        # Return the signal with the same index and columns as close
        return signal.where(~signal.isna(), 0.0)