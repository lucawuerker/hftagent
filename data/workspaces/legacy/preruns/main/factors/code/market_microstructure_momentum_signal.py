"""Market microstructure reveals that stocks experiencing persistent, positive order flow tend to maintain upward momentum due to the influence of liquidity providers and the positive feedback loop created by trend-following traders. By identifying stocks with strong recent order flow and positive price movement, we can exploit these behavioral biases and the amplification of trends in bullish market conditions, as discussed in the findings of Singha (2025).

This factor computes a momentum score based on the cumulative positive order flow over the last 10-second bars, normalized by the stock's price volatility, creating a standardized signal that reflects the strength of upward price trends driven by order flow dynamics.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, stddev, sign, delay, rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class MarketMicrostructureMomentumSignal(BaseFactor):
    factor_id = "market_microstructure_momentum_signal"
    name = "Market Microstructure Momentum Signal"
    category = "momentum"
    inputs = ["orderFlow", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        close = data["close"]
        volatility = stddev(close, 10).replace(0, np.nan)

        cumulative_order_flow = ts_sum(order_flow.where(order_flow > 0, 0), 10)
        momentum_signal = cumulative_order_flow / volatility

        return rank(momentum_signal)  # Normalize the signal to a rank between 0 and 1