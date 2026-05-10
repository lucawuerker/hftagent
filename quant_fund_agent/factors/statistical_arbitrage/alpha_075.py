"""Alpha#75: rank(corr(vwap, volume, 4)) < rank(corr(rank(low), rank(adv50), 12))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, correlation, rank, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha075(BaseFactor):
    factor_id = "alpha_075"
    name = "Alpha#75"
    category = "statistical_arbitrage"
    description = (
        "Binary signal: 1 when ranked 4-day VWAP-volume correlation "
        "is below ranked 12-day low-rank vs ADV50-rank correlation.  "
        "Long when VWAP-volume linkage weakens relative to "
        "low-liquidity co-movement."
    )
    window_length = 62
    inputs = ["close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        v = vwap(data)
        volume = data["volume"]
        adv50 = adv(volume, 50)

        lhs = rank(correlation(v, volume, 4))
        rhs = rank(correlation(rank(data["low"]), rank(adv50), 12))
        return (lhs < rhs).astype(float)
