"""Alpha#95: rank(open - ts_min(open, 12)) < Ts_Rank(rank(corr(sum((high+low)/2, 19), sum(adv40, 19), 13))^5, 12)"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, correlation, rank, ts_min, ts_rank, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha095(BaseFactor):
    factor_id = "alpha_095"
    name = "Alpha#95"
    category = "momentum"
    description = (
        "Binary signal: ranked open distance from 12-day low compared "
        "against ts-ranked 5th-power of ranked midprice-ADV40 "
        "cumulative correlation."
    )
    window_length = 51
    inputs = ["open", "close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]
        adv40 = adv(volume, 40)
        mid = (high + low) / 2.0

        lhs = rank(open_ - ts_min(open_, 12))
        inner = rank(correlation(ts_sum(mid, 19), ts_sum(adv40, 19), 13)) ** 5
        rhs = ts_rank(inner, 12)
        return (lhs < rhs).astype(float)
