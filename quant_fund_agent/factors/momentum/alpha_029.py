"""Alpha#29: min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1*rank(delta((close-1),5))))),2),1))))),1),5) + ts_rank(delay((-1*returns),6),5)

Simplified: the deeply nested left term collapses (product(x,1) = x,
sum(x,1) = x) to ts_min(rank(rank(scale(log(ts_min(rank(rank(-rank(delta(close-1,5)))),2))))),5).
"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (
    delay,
    delta,
    log,
    rank,
    returns,
    scale,
    ts_min,
    ts_rank,
)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha029(BaseFactor):
    factor_id = "alpha_029"
    name = "Alpha#29"
    category = "momentum"
    description = (
        "Complex nested signal combining a scaled log-transform of "
        "deeply ranked close-level changes with a 5-day time-series "
        "rank of 6-day-lagged negative returns."
    )
    window_length = 12
    inputs = ["close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        ret = returns(data)

        inner = -1.0 * rank(delta(close - 1.0, 5))
        inner = rank(rank(inner))
        inner = ts_min(inner, 2)
        inner = log(inner)
        inner = scale(inner)
        inner = rank(rank(inner))
        inner = ts_min(inner, 5)

        rhs = ts_rank(delay(-1.0 * ret, 6), 5)
        return inner + rhs
