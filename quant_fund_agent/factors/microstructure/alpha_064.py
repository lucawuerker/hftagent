"""Alpha#64: (rank(corr(sum(open*0.178+low*0.822, 13), sum(adv120, 13), 17)) < rank(delta((high+low)/2*0.178+vwap*0.822, 4))) * -1"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, correlation, delta, rank, ts_sum, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha064(BaseFactor):
    factor_id = "alpha_064"
    name = "Alpha#64"
    category = "microstructure"
    description = (
        "Negative binary signal: ranked 17-day correlation of blended "
        "open-low sum with cumulative ADV120, compared against ranked "
        "4-day change of a VWAP-midprice blend."
    )
    window_length = 133
    inputs = ["open", "close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        v = vwap(data)
        adv120 = adv(data["volume"], 120)

        blend_a = open_ * 0.178404 + low * (1.0 - 0.178404)
        blend_b = ((high + low) / 2.0) * 0.178404 + v * (1.0 - 0.178404)

        lhs = rank(correlation(ts_sum(blend_a, 13), ts_sum(adv120, 13), 17))
        rhs = rank(delta(blend_b, 4))
        return -1.0 * (lhs < rhs).astype(float)
