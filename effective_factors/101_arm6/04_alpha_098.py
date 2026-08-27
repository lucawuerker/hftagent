"""Alpha#98: rank(decay_linear(corr(vwap, sum(adv5, 26), 5), 7)) - rank(decay_linear(Ts_Rank(Ts_ArgMin(corr(rank(open), rank(adv15), 21), 9), 7), 8))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (
    adv,
    correlation,
    decay_linear,
    rank,
    ts_argmin,
    ts_rank,
    ts_sum,
    vwap,
)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha098(BaseFactor):
    prediction_horizon = 6
    factor_id = "alpha_098"
    name = "Alpha#98"
    category = "statistical_arbitrage"
    description = (
        "Ranked decay-linear VWAP vs cumulative-ADV5 correlation "
        "minus ranked decay-linear of ts-ranked argmin of open-rank "
        "vs ADV15-rank correlation."
    )
    window_length = 55
    inputs = ["open", "close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        volume = data["volume"]
        v = vwap(data)
        adv5 = adv(volume, 5)
        adv15 = adv(volume, 15)

        a = rank(decay_linear(correlation(v, ts_sum(adv5, 26), 5), 7))
        inner = ts_rank(ts_argmin(correlation(rank(open_), rank(adv15), 21), 9), 7)
        b = rank(decay_linear(inner, 8))
        return a - b
