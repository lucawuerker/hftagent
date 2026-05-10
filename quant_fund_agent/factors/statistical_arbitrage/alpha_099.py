"""Alpha#99: (rank(corr(sum((high+low)/2, 20), sum(adv60, 20), 9)) < rank(corr(low, volume, 6))) * -1"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, correlation, rank, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha099(BaseFactor):
    factor_id = "alpha_099"
    name = "Alpha#99"
    category = "statistical_arbitrage"
    description = (
        "Negative binary signal: ranked 9-day correlation of "
        "cumulative midprice with cumulative ADV60, compared "
        "against ranked 6-day low-volume correlation."
    )
    window_length = 80
    inputs = ["close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        volume = data["volume"]
        adv60 = adv(volume, 60)
        mid = (high + low) / 2.0

        lhs = rank(correlation(ts_sum(mid, 20), ts_sum(adv60, 20), 9))
        rhs = rank(correlation(low, volume, 6))
        return -1.0 * (lhs < rhs).astype(float)
