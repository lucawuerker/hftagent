"""Alpha#88: min(rank(decay_linear((rank(open)+rank(low))-(rank(high)+rank(close)), 8)), Ts_Rank(decay_linear(corr(Ts_Rank(close,8), Ts_Rank(adv60,21), 8), 7), 3))"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, correlation, decay_linear, rank, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha088(BaseFactor):
    factor_id = "alpha_088"
    name = "Alpha#88"
    category = "mean_reversion"
    description = (
        "Min of ranked decay-linear OHLC rank-balance (open+low ranks "
        "vs high+close ranks) and ts-ranked decay-linear close-ADV60 "
        "ts-rank correlation.  Picks the weaker of two reversion "
        "signals."
    )
    window_length = 68
    inputs = ["open", "close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]
        adv60 = adv(volume, 60)

        balance = (rank(open_) + rank(low)) - (rank(high) + rank(close))
        a = rank(decay_linear(balance, 8))
        b = ts_rank(
            decay_linear(correlation(ts_rank(close, 8), ts_rank(adv60, 21), 8), 7),
            3,
        )
        return np.minimum(a, b)
