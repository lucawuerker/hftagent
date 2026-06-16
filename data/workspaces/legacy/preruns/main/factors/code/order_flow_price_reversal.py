"""This factor is based on the idea that extreme order flow (either aggressive buying or selling) often precedes price reversals, especially in volatile markets. When large buying or selling pressure occurs, it can push prices away from their equilibrium state, creating an opportunity for mean reversion as prices tend to revert back to their mean after such extreme movements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (rank, delta, ts_sum, ts_mean, stddev, sign, abs_,)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class OrderFlowPriceReversal(BaseFactor):
    factor_id = "order_flow_price_reversal"
    name = "Order Flow Price Reversal"
    category = "mean_reversion"
    inputs = ["orderFlow", "close"]
    description = "The factor computes the z-score of the order flow in relation to the price movement over the last 10 seconds. A significant positive or negative z-score indicates extreme buying or selling pressure, suggesting a potential price reversal."

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        close = data["close"]

        # Calculate the price movement over the last 10 seconds
        price_movement = delta(close, 1)

        # Calculate mean and stddev of order flow and price movement over the last 10 seconds
        mean_order_flow = ts_mean(order_flow, 10)
        std_order_flow = stddev(order_flow, 10)
        mean_price_movement = ts_mean(price_movement, 10)
        std_price_movement = stddev(price_movement, 10)

        # Calculate z-scores
        z_order_flow = (order_flow - mean_order_flow) / std_order_flow.replace(0, np.nan)
        z_price_movement = (price_movement - mean_price_movement) / std_price_movement.replace(0, np.nan)

        # Combine z-scores
        combined_z_score = z_order_flow + z_price_movement

        return combined_z_score