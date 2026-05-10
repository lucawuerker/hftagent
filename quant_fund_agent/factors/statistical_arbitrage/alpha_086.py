"""Alpha#86: (Ts_Rank(corr(close, sum(adv20, 15), 6), 20) < rank((open+close) - (vwap+open))) * -1

The RHS simplifies to rank(close - vwap).
"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, correlation, rank, ts_rank, ts_sum, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha086(BaseFactor):
    factor_id = "alpha_086"
    name = "Alpha#86"
    category = "statistical_arbitrage"
    description = (
        "Negative binary signal: ts-ranked close vs cumulative-ADV20 "
        "correlation (20-day) compared against ranked close-VWAP "
        "deviation."
    )
    window_length = 35
    inputs = ["close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        v = vwap(data)
        adv20 = adv(volume, 20)

        lhs = ts_rank(correlation(close, ts_sum(adv20, 15), 6), 20)
        rhs = rank(close - v)
        return -1.0 * (lhs < rhs).astype(float)
