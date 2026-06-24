"""According to the paper by Easley et al. (2014), order flow imbalance can indicate significant price movement due to information asymmetries and market toxicity. A persistent imbalance in buy or sell orders suggests that market participants expect price changes, thus providing a predictive signal for future returns. This factor captures the net order flow, allowing traders to identify potential price trends before they fully materialize."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, sign, adv
from quant_fund_agent.factors.registry import register_factor


@register_factor
class OrderFlowImbalanceSignal(BaseFactor):
    factor_id = "order_flow_imbalance_signal"
    name = "Order Flow Imbalance Signal"
    category = "microstructure"
    description = "This signal computes the difference between buy and sell volume over a specified period, normalized by the total volume. A positive value indicates a bullish order flow, while a negative value indicates bearish sentiment."
    inputs = ["volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"].fillna(0.0)
        buy_volume = volume.where(volume > 0, 0.0)
        sell_volume = volume.where(volume < 0, 0.0).abs()
        net_order_flow = ts_sum(buy_volume, 5) - ts_sum(sell_volume, 5)
        total_volume = ts_sum(volume.abs(), 5)
        imbalance_signal = net_order_flow / total_volume.replace(0, np.nan)
        return imbalance_signal