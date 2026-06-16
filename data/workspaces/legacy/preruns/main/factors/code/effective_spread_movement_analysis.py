"""In financial markets, price movements often reflect underlying supply and demand dynamics. By analyzing the changes in effective spread over time, we can identify shifts in market sentiment and liquidity conditions, potentially signaling short-term price reversals or continuations. This aligns with the findings in the paper 'DeepVol: Volatility Forecasting from High-Frequency Data with Dilated Causal Convolutions', which highlights the importance of intra-bar dynamics.

This factor computes the rolling average and standard deviation of the effective spread across a defined lookback period, highlighting sudden movements that may indicate shifts in market conditions or liquidity pressures.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_mean, stddev

@register_factor
class EffectiveSpreadMovementAnalysis(BaseFactor):
    factor_id = "effective_spread_movement_analysis"
    name = "Effective Spread Movement Analysis"
    category = "microstructure"
    inputs = ["effSpread"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        eff_spread = data["effSpread"].fillna(0.0)
        mean_eff_spread = ts_mean(eff_spread, 60)
        std_eff_spread = stddev(eff_spread, 60)
        return mean_eff_spread + std_eff_spread