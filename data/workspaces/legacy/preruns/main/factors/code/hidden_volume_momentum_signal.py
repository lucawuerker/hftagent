"""Hidden volume, often associated with informed trading, can indicate the direction of future price movements. When hidden volume increases alongside price momentum, it suggests that informed traders are accumulating positions, potentially leading to significant upward price movements. This aligns with the idea that higher hidden volume can precede bullish trends due to informed buying activity."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (rank, delta, ts_sum, correlation, returns)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class HiddenVolumeMomentumSignal(BaseFactor):
    factor_id = "hidden_volume_momentum_signal"
    name = "Hidden Volume Momentum Signal"
    category = "momentum"
    inputs = ["hidden", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        hidden_volume = data["hidden"].fillna(0.0)
        close = data["close"]

        # Calculate the rate of change of hidden volume and price returns
        hidden_volume_change = delta(hidden_volume, 1)
        price_returns = returns(data)

        # Calculate the correlation between hidden volume change and price returns over a 60-bar window
        correlation_signal = correlation(hidden_volume_change, price_returns, 60)

        # Rank the correlation signal to normalize it
        ranked_signal = rank(correlation_signal)

        return ranked_signal