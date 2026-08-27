"""Alpha#91: ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close, IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)

Fractional windows rounded to integers: corr 10, decay 16, decay 4,
ts_rank 5; corr 4, decay 3.  Requires ``data["industry"]`` (falls back
to sector).
"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors._labels import neutralize
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
class Alpha091(BaseFactor):
    prediction_horizon = 6
    factor_id = "alpha_091"
    name = "Alpha#91"
    category = "statistical_arbitrage"
    description = (
        "Negative spread between a doubly decay-smoothed ts-ranked "
        "correlation of industry-neutralized close with volume and the "
        "ranked decay-smoothed vwap-ADV30 correlation.  Requires "
        "data['industry']."
    )
    window_length = 70
    inputs = ["high", "low", "close", "volume", "industry"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        v = vwap(data)
        adv30 = adv(volume, 30)

        close_neut = neutralize(close, data, "industry", "Alpha#91")
        term1 = ts_rank(
            decay_linear(decay_linear(correlation(close_neut, volume, 10), 16), 4), 5
        )

        term2 = rank(decay_linear(correlation(v, adv30, 4), 3))
        return -1.0 * (term1 - term2)
