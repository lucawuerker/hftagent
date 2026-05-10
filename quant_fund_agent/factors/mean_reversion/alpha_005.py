"""Alpha#5: rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap))))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, rank, ts_sum, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha005(BaseFactor):
    factor_id = "alpha_005"
    name = "Alpha#5"
    category = "mean_reversion"
    description = (
        "Interaction between open-price deviation from 10-day VWAP mean "
        "and close-to-VWAP rank.  Captures mean-reversion around the "
        "volume-weighted fair value."
    )
    window_length = 10
    inputs = ["open", "close", "high", "low"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        close = data["close"]
        v = vwap(data)
        lhs = rank(open_ - (ts_sum(v, 10) / 10.0))
        rhs = -1.0 * abs_(rank(close - v))
        return lhs * rhs
