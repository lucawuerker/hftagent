"""Alpha#81: (rank(log(product(rank(rank(corr(vwap, sum(adv10,50), 8))^4), 15))) < rank(corr(rank(vwap), rank(volume), 5))) * -1"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (
    adv,
    correlation,
    log,
    product,
    rank,
    ts_sum,
    vwap,
)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha081(BaseFactor):
    factor_id = "alpha_081"
    name = "Alpha#81"
    category = "statistical_arbitrage"
    description = (
        "Negative binary signal: ranked log of rolling product of "
        "4th-power ranked VWAP-cumADV10 correlation, compared against "
        "ranked VWAP-rank vs volume-rank correlation."
    )
    window_length = 65
    inputs = ["close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"]
        v = vwap(data)
        adv10 = adv(volume, 10)

        inner = rank(rank(correlation(v, ts_sum(adv10, 50), 8))) ** 4
        lhs = rank(log(product(rank(inner), 15)))
        rhs = rank(correlation(rank(v), rank(volume), 5))
        return -1.0 * (lhs < rhs).astype(float)
