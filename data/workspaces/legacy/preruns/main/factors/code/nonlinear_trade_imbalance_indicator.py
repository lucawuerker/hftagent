"""Research indicates a nonlinear relationship between trade imbalance and price changes, suggesting that extreme values of trade imbalance may lead to disproportionate price impacts. This is particularly relevant in a high-frequency trading context, where rapid changes in order flow can lead to significant price movements. By capturing these nonlinearities, traders can better predict price movements based on the current trade imbalance."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, signed_power, scale, abs_, delay

@register_factor
class NonlinearTradeImbalanceIndicator(BaseFactor):
    factor_id = "nonlinear_trade_imbalance_indicator"
    name = "Nonlinear Trade Imbalance Indicator"
    category = "microstructure"
    description = "Computes a nonlinear transformation of trade imbalance to reflect expected price impact."
    inputs = ["trade", "orderFlow"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        trade = data["trade"].fillna(0.0)
        order_flow = data["orderFlow"].fillna(0.0)

        # Calculate trade imbalance
        trade_imbalance = trade / (trade.abs() + order_flow.abs()).replace(0, np.nan)
        trade_imbalance = trade_imbalance.fillna(0.0)

        # Apply a nonlinear transformation (e.g., logistic function)
        nonlinear_signal = signed_power(trade_imbalance, 3)  # Emphasizing extremes

        # Scale the output to ensure it is within a reasonable range
        scaled_signal = scale(nonlinear_signal)

        return scaled_signal