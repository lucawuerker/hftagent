"""Alpha#61: rank(vwap - ts_min(vwap, 16)) < rank(correlation(vwap, adv180, 18))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, correlation, rank, ts_min, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha061(BaseFactor):
    factor_id = "alpha_061"
    name = "Alpha#61"
    category = "mean_reversion"
    description = (
        "Binary signal: 1 when ranked VWAP distance from 16-day low "
        "is less than ranked 18-day VWAP-ADV180 correlation, else 0.  "
        "Long when VWAP is near its floor relative to its volume "
        "co-movement."
    )
    window_length = 180
    inputs = ["close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        v = vwap(data)
        adv180 = adv(data["volume"], 180)
        lhs = rank(v - ts_min(v, 16))
        rhs = rank(correlation(v, adv180, 18))
        return (lhs < rhs).astype(float)
