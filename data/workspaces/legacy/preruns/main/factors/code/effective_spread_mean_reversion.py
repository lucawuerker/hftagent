"""Effective spreads tend to revert to their mean over time as market participants adjust their orders based on recent price movements and liquidity conditions. This behavior is grounded in the notion that extreme spreads during periods of high volatility or low liquidity signal a potential return to more normalized conditions, thus providing a trading opportunity for mean reversion strategies."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_mean, ts_sum, ts_rank, abs_, delay
from quant_fund_agent.factors.registry import register_factor


@register_factor
class EffectiveSpreadMeanReversion(BaseFactor):
    factor_id = "effective_spread_mean_reversion"
    name = "Effective Spread Mean Reversion"
    category = "mean_reversion"
    inputs = ["effSpread"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        eff_spread = data["effSpread"].fillna(0.0)
        mean_eff_spread = ts_mean(eff_spread, 60)
        deviation = eff_spread - mean_eff_spread
        return ts_rank(deviation, 60)