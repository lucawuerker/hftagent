"""Alpha#68: (Ts_Rank(corr(rank(high), rank(adv15), 9), 14) < rank(delta(close*0.518+low*0.482, 1))) * -1"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, correlation, delta, rank, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha068(BaseFactor):
    factor_id = "alpha_068"
    name = "Alpha#68"
    category = "momentum"
    description = (
        "Negative binary signal: ts-ranked 9-day high-ADV15 "
        "correlation (14-day window) compared against ranked 1-day "
        "change of a close-low blend."
    )
    window_length = 23
    inputs = ["close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"]
        close = data["close"]
        low = data["low"]
        adv15 = adv(data["volume"], 15)

        blend = close * 0.518371 + low * (1.0 - 0.518371)
        lhs = ts_rank(correlation(rank(high), rank(adv15), 9), 14)
        rhs = rank(delta(blend, 1))
        return -1.0 * (lhs < rhs).astype(float)
