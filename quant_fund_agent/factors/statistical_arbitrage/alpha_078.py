"""Alpha#78: rank(corr(sum(low*0.352+vwap*0.648, 20), sum(adv40, 20), 7)) ^ rank(corr(rank(vwap), rank(volume), 6))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, correlation, rank, ts_sum, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha078(BaseFactor):
    factor_id = "alpha_078"
    name = "Alpha#78"
    category = "statistical_arbitrage"
    description = (
        "Ranked 7-day correlation of 20-day cumulative low-VWAP blend "
        "with cumulative ADV40, raised to the power of ranked 6-day "
        "VWAP-rank vs volume-rank correlation."
    )
    window_length = 47
    inputs = ["close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        low = data["low"]
        volume = data["volume"]
        v = vwap(data)
        adv40 = adv(volume, 40)

        blend = low * 0.352233 + v * (1.0 - 0.352233)
        base = rank(correlation(ts_sum(blend, 20), ts_sum(adv40, 20), 7))
        exp = rank(correlation(rank(v), rank(volume), 6))
        return base ** exp
