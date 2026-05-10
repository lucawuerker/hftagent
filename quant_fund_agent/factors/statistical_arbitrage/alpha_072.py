"""Alpha#72: rank(decay_linear(corr((high+low)/2, adv40, 9), 10)) / rank(decay_linear(corr(Ts_Rank(vwap,4), Ts_Rank(volume,19), 7), 3))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (
    adv,
    correlation,
    decay_linear,
    rank,
    ts_rank,
    vwap,
)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha072(BaseFactor):
    factor_id = "alpha_072"
    name = "Alpha#72"
    category = "statistical_arbitrage"
    description = (
        "Ratio of ranked decay-linear midprice-ADV40 correlation to "
        "ranked decay-linear VWAP-ts-rank / volume-ts-rank correlation.  "
        "Captures relative strength of midprice-liquidity vs "
        "VWAP-volume co-movement."
    )
    window_length = 59
    inputs = ["close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        volume = data["volume"]
        v = vwap(data)
        adv40 = adv(volume, 40)
        mid = (high + low) / 2.0

        num = rank(decay_linear(correlation(mid, adv40, 9), 10))
        den = rank(
            decay_linear(correlation(ts_rank(v, 4), ts_rank(volume, 19), 7), 3)
        ).replace(0, float("nan"))
        return num / den
