"""Alpha#92: min(Ts_Rank(decay_linear(((high+low)/2+close) < (low+open), 15), 19), Ts_Rank(decay_linear(corr(rank(low), rank(adv30), 8), 7), 7))"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, correlation, decay_linear, rank, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha092(BaseFactor):
    factor_id = "alpha_092"
    name = "Alpha#92"
    category = "mean_reversion"
    description = (
        "Min of two ts-ranked decay-linear signals: (1) boolean "
        "midprice+close < low+open condition, and (2) ranked-low vs "
        "ranked-ADV30 correlation.  Picks the weaker reversion signal."
    )
    window_length = 49
    inputs = ["open", "close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]
        adv30 = adv(volume, 30)

        cond = (((high + low) / 2.0 + close) < (low + open_)).astype(float)
        a = ts_rank(decay_linear(cond, 15), 19)
        b = ts_rank(decay_linear(correlation(rank(low), rank(adv30), 8), 7), 7)
        return np.minimum(a, b)
