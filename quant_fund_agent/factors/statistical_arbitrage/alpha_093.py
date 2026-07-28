"""Alpha#93: (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81, 17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 2.77377), 16.2664)))

Fractional windows rounded to integers: corr 17, decay 20, ts_rank 8;
delta 3, decay 16.  Requires ``data["industry"]`` (falls back to
sector).
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
class Alpha093(BaseFactor):
    factor_id = "alpha_093"
    name = "Alpha#93"
    category = "statistical_arbitrage"
    description = (
        "Ts-ranked decay-smoothed correlation of industry-neutralized "
        "VWAP with ADV81 divided by the ranked decay-linear 3-day "
        "change of a close-vwap blend.  Requires data['industry']."
    )
    window_length = 130
    inputs = ["high", "low", "close", "volume", "industry"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        v = vwap(data)
        adv81 = adv(data["volume"], 81)

        v_neut = neutralize(v, data, "industry", "Alpha#93")
        numerator = ts_rank(decay_linear(correlation(v_neut, adv81, 17), 20), 8)

        blend = close * 0.524434 + v * (1.0 - 0.524434)
        denominator = rank(decay_linear(delta(blend, 3), 16))
        return numerator / denominator
