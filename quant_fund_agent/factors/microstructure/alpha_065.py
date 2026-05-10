"""Alpha#65: (rank(corr(open*0.008+vwap*0.992, sum(adv60, 9), 6)) < rank(open - ts_min(open, 14))) * -1"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, correlation, rank, ts_min, ts_sum, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha065(BaseFactor):
    factor_id = "alpha_065"
    name = "Alpha#65"
    category = "microstructure"
    description = (
        "Negative binary signal: ranked 6-day correlation of near-VWAP "
        "blend with cumulative ADV60, compared against ranked open "
        "distance from 14-day low."
    )
    window_length = 69
    inputs = ["open", "close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        v = vwap(data)
        adv60 = adv(data["volume"], 60)

        blend = open_ * 0.00817205 + v * (1.0 - 0.00817205)
        lhs = rank(correlation(blend, ts_sum(adv60, 9), 6))
        rhs = rank(open_ - ts_min(open_, 14))
        return -1.0 * (lhs < rhs).astype(float)
