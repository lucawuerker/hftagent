"""Markets often exhibit mean-reverting behavior following extreme trends, particularly when driven by order flow imbalances. When rapid buying or selling occurs, it can lead to short-term price movements that are unsustainable, creating opportunities for reversal trades as the market corrects. This factor aims to identify stocks where recent order flow momentum diverges significantly from price movement, indicating potential reversal points. This signal computes the difference between the recent order flow (buy vs. sell) momentum and the price momentum, highlighting discrepancies that may precede reversals. A significant divergence suggests that the current price trend is likely to reverse in the near future."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (delta, ts_sum, ts_mean, ts_rank, sign)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class OrderFlowMomentumReversalSignal(BaseFactor):
    factor_id = "order_flow_momentum_reversal_signal"
    name = "Order Flow Momentum Reversal Signal"
    category = "momentum"
    inputs = ["orderFlow", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        close = data["close"]

        # Calculate price momentum
        price_momentum = delta(close, 5)

        # Calculate order flow momentum
        order_flow_momentum = delta(order_flow, 5)

        # Calculate the divergence
        divergence = order_flow_momentum - price_momentum

        # Normalize the divergence
        normalized_divergence = ts_rank(divergence, 60).fillna(0.0)

        return normalized_divergence