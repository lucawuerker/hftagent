"""Alpha#74: (rank(corr(close, sum(adv30, 37), 15)) < rank(corr(rank(high*0.026+vwap*0.974), rank(volume), 11))) * -1"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, correlation, rank, ts_sum, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha074(BaseFactor):
    factor_id = "alpha_074"
    name = "Alpha#74"
    category = "statistical_arbitrage"
    description = (
        "Negative binary signal: ranked close vs cumulative-ADV30 "
        "correlation compared against ranked near-VWAP-blend vs "
        "volume-rank correlation."
    )
    window_length = 67
    inputs = ["close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        volume = data["volume"]
        v = vwap(data)
        adv30 = adv(volume, 30)

        blend = high * 0.0261661 + v * (1.0 - 0.0261661)
        lhs = rank(correlation(close, ts_sum(adv30, 37), 15))
        rhs = rank(correlation(rank(blend), rank(volume), 11))
        return -1.0 * (lhs < rhs).astype(float)
