"""Heuristic-driven traders tend to exhibit contrarian behavior, buying after losses and selling after gains due to loss aversion and mental accounting biases. This trait can create significant price reversals, particularly in volatile markets, as these traders react to past price movements rather than fundamentals. Understanding this behavior can help identify potential entry and exit points based on order flow dynamics."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (rank, delta, ts_sum, ts_mean, adv, returns)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class HeuristicTraderContrarianSignal(BaseFactor):
    factor_id = "heuristic_trader_contrarian_signal"
    name = "Heuristic Trader Contrarian Signal"
    category = "microstructure"
    description = "This signal calculates the expected price movement based on the previous day’s returns and the volume traded, identifying periods where contrarian trading activity is likely to influence price reversals."
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        returns_prev = returns(data).fillna(0.0)
        adv_volume = adv(volume, 20)

        contrarian_signal = ts_sum(returns_prev, 5) * ts_mean(volume, 5)
        return contrarian_signal.where(adv_volume < volume, 0.0)