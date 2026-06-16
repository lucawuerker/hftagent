"""The effective spread reflects the true cost of trading and can indicate the market's liquidity. A rising effective spread may suggest increasing transaction costs and reduced liquidity, which could precede price corrections. Conversely, a declining effective spread may indicate emerging momentum in a stock as liquidity improves, aligning with behavioral finance theories about market reaction to liquidity changes."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, delta, ts_rank

@register_factor
class EffectiveSpreadMomentum(BaseFactor):
    factor_id = "effective_spread_momentum"
    name = "Effective Spread Momentum"
    category = "momentum"
    description = ("This factor computes the change in the effective spread over a rolling window, identifying stocks with improving or deteriorating liquidity conditions. "
                   "It tracks this momentum to signal potential entry or exit points.")
    window_length = 5
    inputs = ["effSpread"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        eff_spread = data["effSpread"].fillna(0.0)
        spread_change = delta(eff_spread, 1)
        momentum_signal = ts_rank(spread_change, self.window_length)
        return momentum_signal