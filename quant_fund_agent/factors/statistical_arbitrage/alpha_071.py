"""Alpha#71: max(Ts_Rank(decay_linear(corr(Ts_Rank(close,3),Ts_Rank(adv180,12),18),4),16), Ts_Rank(decay_linear(rank((low+open-2*vwap)^2),16),4))"""

from __future__ import annotations

import numpy as np
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
class Alpha071(BaseFactor):
    factor_id = "alpha_071"
    name = "Alpha#71"
    category = "statistical_arbitrage"
    description = (
        "Max of two ts-ranked decay-linear signals: (1) correlation "
        "of close ts-rank with ADV180 ts-rank, and (2) squared ranked "
        "deviation of (low+open) from 2×VWAP."
    )
    window_length = 196
    inputs = ["close", "open", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        open_ = data["open"]
        low = data["low"]
        volume = data["volume"]
        v = vwap(data)
        adv180 = adv(volume, 180)

        a = ts_rank(
            decay_linear(correlation(ts_rank(close, 3), ts_rank(adv180, 12), 18), 4),
            16,
        )
        b = ts_rank(
            decay_linear(rank((low + open_ - v - v) ** 2), 16),
            4,
        )
        return np.maximum(a, b)
