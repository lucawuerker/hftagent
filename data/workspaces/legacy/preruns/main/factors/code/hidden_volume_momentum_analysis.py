"""The momentum of hidden volume can provide insights into future price movements, as significant hidden orders often indicate strong underlying demand or supply that is not immediately visible in the market. This aligns with the idea that large, unexecuted orders can influence price direction when they eventually become visible or are executed."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, delta, ts_rank, abs_, sign
from quant_fund_agent.factors.registry import register_factor


@register_factor
class HiddenVolumeMomentumAnalysis(BaseFactor):
    factor_id = "hidden_volume_momentum_analysis"
    name = "Hidden Volume Momentum Analysis"
    category = "momentum"
    description = "This factor computes the momentum of hidden volume over a specified lookback period, comparing it to the price change to identify potential price movements driven by underlying demand or supply imbalances."
    inputs = ["hidden", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        hidden_volume = data["hidden"].fillna(0.0)
        close = data["close"]
        price_change = delta(close, 1).fillna(0.0)
        momentum_hidden_volume = ts_sum(hidden_volume, 60)
        momentum_signal = ts_rank(momentum_hidden_volume, 60) * sign(price_change)
        return momentum_signal