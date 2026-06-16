"""As markets evolve, the effective spread can reflect changing liquidity conditions and trader sentiment. By analyzing momentum in effective spread values over time, traders can identify periods of increasing or decreasing liquidity, which often precede significant price movements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_mean, delta, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class EffectiveSpreadMomentumAnalysis(BaseFactor):
    factor_id = "effective_spread_momentum_analysis"
    name = "Effective Spread Momentum Analysis"
    category = "microstructure"
    inputs = ["effSpread"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        eff_spread = data["effSpread"].fillna(0.0)
        momentum = delta(eff_spread, 5)
        signal = ts_rank(momentum, 60)
        return signal