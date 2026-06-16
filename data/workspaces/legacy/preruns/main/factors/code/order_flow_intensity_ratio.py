"""Increased order flow intensity can indicate the strength of market participants' conviction, especially when comparing buyer-initiated and seller-initiated trades. By analyzing the ratio of buyer-initiated to seller-initiated order flow, traders can identify potential trends or reversals, as stronger buying pressure often leads to upward price movement, while stronger selling pressure can indicate downtrends.

This factor computes the ratio of signed trade volume (order flow) across a specified period, normalizing it to capture the intensity of buyer-initiated versus seller-initiated trades. A higher ratio suggests a bullish sentiment, while a lower ratio indicates bearish sentiment."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, delta, abs_, sign


@register_factor
class OrderFlowIntensityRatio(BaseFactor):
    factor_id = "order_flow_intensity_ratio"
    name = "Order Flow Intensity Ratio"
    category = "microstructure"
    description = "This factor computes the ratio of signed trade volume (order flow) across a specified period, normalizing it to capture the intensity of buyer-initiated versus seller-initiated trades. A higher ratio suggests a bullish sentiment, while a lower ratio indicates bearish sentiment."
    inputs = ["trade"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        trade = data["trade"].fillna(0.0)
        buyer_initiated = trade.where(trade > 0, 0.0)
        seller_initiated = -trade.where(trade < 0, 0.0)

        sum_buyer = ts_sum(buyer_initiated, 5)
        sum_seller = ts_sum(seller_initiated, 5)

        ratio = sum_buyer / sum_seller.replace(0, np.nan)
        return ratio.fillna(0.0)