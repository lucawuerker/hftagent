"""Alpha#96: max(Ts_Rank(decay_linear(corr(rank(vwap),rank(volume),4),5),8), Ts_Rank(decay_linear(Ts_ArgMax(corr(Ts_Rank(close,7),Ts_Rank(adv60,4),4),13),14),13)) * -1"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (
    adv,
    correlation,
    decay_linear,
    rank,
    ts_argmax,
    ts_rank,
    vwap,
)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha096(BaseFactor):
    factor_id = "alpha_096"
    name = "Alpha#96"
    category = "statistical_arbitrage"
    description = (
        "Negative max of two ts-ranked decay-linear signals: "
        "(1) VWAP-rank vs volume-rank correlation, and (2) argmax of "
        "close-ts-rank vs ADV60-ts-rank correlation."
    )
    window_length = 74
    inputs = ["close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        v = vwap(data)
        adv60 = adv(volume, 60)

        a = ts_rank(
            decay_linear(correlation(rank(v), rank(volume), 4), 5),
            8,
        )
        inner_corr = correlation(ts_rank(close, 7), ts_rank(adv60, 4), 4)
        b = ts_rank(
            decay_linear(ts_argmax(inner_corr, 13), 14),
            13,
        )
        return -1.0 * np.maximum(a, b)
