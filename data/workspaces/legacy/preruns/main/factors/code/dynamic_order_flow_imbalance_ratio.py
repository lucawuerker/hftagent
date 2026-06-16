"""Order flow imbalance is a strong indicator of future price movement, as it reflects the relative buying and selling pressure in the market. By analyzing the ratio of aggressive buyer-initiated trades to aggressive seller-initiated trades over a rolling window, this factor captures shifts in market sentiment and liquidity, allowing traders to anticipate potential price movements. Research indicates that this imbalance can predict short-term price movements due to liquidity constraints and market impact effects."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, delta, abs_, sign


@register_factor
class DynamicOrderFlowImbalanceRatio(BaseFactor):
    factor_id = "dynamic_order_flow_imbalance_ratio"
    name = "Dynamic Order Flow Imbalance Ratio"
    category = "microstructure"
    description = "This factor computes the ratio of the volume of buyer-initiated trades to seller-initiated trades over a specified rolling window, adjusted for total trade volume to normalize the signal. It provides insights into market sentiment and potential price direction based on order flow."
    inputs = ["trade", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        trade = data["trade"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        buyer_initiated = trade.where(trade > 0, 0).fillna(0.0)
        seller_initiated = trade.where(trade < 0, 0).fillna(0.0)

        sum_buyer = ts_sum(buyer_initiated, 5)
        sum_seller = ts_sum(seller_initiated, 5)
        total_volume = ts_sum(volume, 5)

        ratio = sum_buyer / (sum_seller.replace(0, np.nan) + 1e-10)  # Avoid division by zero
        normalized_ratio = ratio / total_volume.replace(0, np.nan)  # Normalize by total volume

        return normalized_ratio.fillna(0.0)