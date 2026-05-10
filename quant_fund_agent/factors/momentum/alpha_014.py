"""Alpha#14: (-1 * rank(delta(returns, 3))) * correlation(open, volume, 10)"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import correlation, delta, rank, returns
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha014(BaseFactor):
    factor_id = "alpha_014"
    name = "Alpha#14"
    category = "momentum"
    description = (
        "Negative ranked 3-day return acceleration scaled by the "
        "10-day open-volume correlation.  Momentum signal weighted "
        "by the strength of the open-volume relationship."
    )
    window_length = 10
    inputs = ["open", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        ret = returns(data)
        lhs = -1.0 * rank(delta(ret, 3))
        rhs = correlation(data["open"], data["volume"], 10)
        return lhs * rhs
