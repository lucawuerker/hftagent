"""Alpha#63: ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180, 37.2467), 13.557), 12.2883))) * -1)

Fractional windows rounded to integers (house convention, cf. Alpha#58):
delta 2, decay 8; sum(adv180, 37), corr 14, decay 12.  Requires
``data["industry"]``; falls back to ``sector``, else skips neutralisation.
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
    ts_sum,
    vwap,
)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha063(BaseFactor):
    factor_id = "alpha_063"
    name = "Alpha#63"
    category = "statistical_arbitrage"
    description = (
        "Negative spread between ranked decay-linear 2-day change of "
        "industry-neutralized close and ranked decay-linear correlation "
        "of a vwap-open blend with smoothed ADV180.  Requires "
        "data['industry'] (falls back to sector)."
    )
    window_length = 250
    inputs = ["open", "high", "low", "close", "volume", "industry"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        close = data["close"]
        v = vwap(data)

        close_neut = neutralize(close, data, "industry", "Alpha#63")
        term1 = rank(decay_linear(delta(close_neut, 2), 8))

        blend = v * 0.318108 + open_ * (1.0 - 0.318108)
        adv_sum = ts_sum(adv(data["volume"], 180), 37)
        term2 = rank(decay_linear(correlation(blend, adv_sum, 14), 12))

        return -1.0 * (term1 - term2)
