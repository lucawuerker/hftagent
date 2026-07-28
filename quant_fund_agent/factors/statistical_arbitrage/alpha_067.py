"""Alpha#67: ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)

Fractional windows rounded to integers: ts_min 2, corr 6.  Requires
``data["sector"]`` and ``data["subindustry"]`` (the latter falls back to
``industry`` then ``sector``); skips neutralisation if labels are absent.
"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors._labels import neutralize
from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, correlation, rank, ts_min, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha067(BaseFactor):
    factor_id = "alpha_067"
    name = "Alpha#67"
    category = "statistical_arbitrage"
    description = (
        "Negative ranked 2-day high breakout raised to the ranked "
        "correlation of sector-neutralized VWAP with "
        "subindustry-neutralized ADV20.  Requires data['sector'] and "
        "data['subindustry']."
    )
    window_length = 30
    inputs = ["high", "low", "close", "volume", "sector", "industry"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"]
        v = vwap(data)
        adv20 = adv(data["volume"], 20)

        v_neut = neutralize(v, data, "sector", "Alpha#67")
        adv_neut = neutralize(adv20, data, "subindustry", "Alpha#67")

        base = rank(high - ts_min(high, 2))
        expo = rank(correlation(v_neut, adv_neut, 6))
        return -1.0 * base**expo
