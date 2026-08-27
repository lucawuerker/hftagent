"""Alpha#97: ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))), IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low, 7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)

Fractional windows rounded to integers: delta 3, decay 20; ts_rank(low)
8, ts_rank(adv60) 17, corr 5, ts_rank 19, decay 16, ts_rank 7.
Requires ``data["industry"]`` (falls back to sector).
"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors._labels import neutralize
from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (
    adv,
    correlation,
    decay_linear,
    delta,
    rank,
    ts_rank,
    vwap,
)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha097(BaseFactor):
    prediction_horizon = 6
    factor_id = "alpha_097"
    name = "Alpha#97"
    category = "statistical_arbitrage"
    description = (
        "Negative spread between the ranked decay-linear 3-day change "
        "of an industry-neutralized low-vwap blend and a deeply "
        "ts-ranked decay-smoothed correlation of ts-ranked low with "
        "ts-ranked ADV60.  Requires data['industry']."
    )
    window_length = 130
    inputs = ["high", "low", "close", "volume", "industry"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        low = data["low"]
        v = vwap(data)
        adv60 = adv(data["volume"], 60)

        blend = low * 0.721001 + v * (1.0 - 0.721001)
        blend_neut = neutralize(blend, data, "industry", "Alpha#97")
        term1 = rank(decay_linear(delta(blend_neut, 3), 20))

        corr = correlation(ts_rank(low, 8), ts_rank(adv60, 17), 5)
        term2 = ts_rank(decay_linear(ts_rank(corr, 19), 16), 7)
        return -1.0 * (term1 - term2)
