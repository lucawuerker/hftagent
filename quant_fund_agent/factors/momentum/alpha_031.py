"""Alpha#31: rank(rank(rank(decay_linear((-1*rank(rank(delta(close,10)))),10)))) + rank((-1*delta(close,3))) + sign(scale(correlation(adv20,low,12)))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (
    adv,
    correlation,
    decay_linear,
    delta,
    rank,
    scale,
    sign,
)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha031(BaseFactor):
    factor_id = "alpha_031"
    name = "Alpha#31"
    category = "momentum"
    description = (
        "Three-term composite: triply-ranked decay-linear of the "
        "10-day close delta, ranked 3-day close reversal, and the "
        "sign of scaled ADV20-low correlation."
    )
    window_length = 20
    inputs = ["close", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        adv20 = adv(volume, 20)

        a = rank(rank(rank(decay_linear(-1.0 * rank(rank(delta(close, 10))), 10))))
        b = rank(-1.0 * delta(close, 3))
        c = sign(scale(correlation(adv20, data["low"], 12)))
        return a + b + c
