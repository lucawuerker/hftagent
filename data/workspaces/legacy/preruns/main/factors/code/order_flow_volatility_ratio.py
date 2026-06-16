"""The relationship between order flow and price volatility can signal impending price movements. A higher ratio of order flow volatility to price volatility suggests that larger order flow fluctuations are likely to lead to significant price changes, driven by market participants reacting to supply and demand imbalances. This aligns with microstructure theories emphasizing the role of order flow in price formation."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import stddev, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class OrderFlowVolatilityRatio(BaseFactor):
    factor_id = "order_flow_volatility_ratio"
    name = "Order Flow Volatility Ratio"
    category = "microstructure"
    inputs = ["close", "orderFlow"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        order_flow = data["orderFlow"].fillna(0.0)

        price_volatility = stddev(close, 10)
        order_flow_volatility = stddev(order_flow, 10)

        # Avoid division by zero by replacing zeros in price_volatility with NaN
        price_volatility = price_volatility.replace(0, np.nan)

        # Calculate the ratio
        ratio = order_flow_volatility / price_volatility

        return ratio