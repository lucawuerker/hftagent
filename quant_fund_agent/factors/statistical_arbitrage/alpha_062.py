"""Alpha#62: (rank(corr(vwap, sum(adv20,22), 10)) < rank((rank(open)+rank(open)) < (rank((high+low)/2)+rank(high)))) * -1"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, correlation, rank, ts_sum, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha062(BaseFactor):
    factor_id = "alpha_062"
    name = "Alpha#62"
    category = "statistical_arbitrage"
    description = (
        "Negative binary signal comparing ranked VWAP-cumulative-ADV20 "
        "correlation with a ranked open vs midprice/high composite "
        "inequality."
    )
    window_length = 32
    inputs = ["open", "close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        v = vwap(data)
        adv20 = adv(data["volume"], 20)
        high = data["high"]
        low = data["low"]
        open_ = data["open"]

        lhs = rank(correlation(v, ts_sum(adv20, 22), 10))
        open_rank = rank(open_) + rank(open_)
        mid_high = rank((high + low) / 2.0) + rank(high)
        rhs = rank((open_rank < mid_high).astype(float))
        return -1.0 * (lhs < rhs).astype(float)
