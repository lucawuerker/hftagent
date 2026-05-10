"""Alpha#36: weighted 5-term composite of correlation, gap, lagged-return rank, VWAP, and trend signals."""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (
    abs_,
    adv,
    correlation,
    delay,
    rank,
    returns,
    ts_rank,
    ts_mean,
    vwap,
)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha036(BaseFactor):
    factor_id = "alpha_036"
    name = "Alpha#36"
    category = "statistical_arbitrage"
    description = (
        "Five-term weighted composite: 15-day gap-volume correlation, "
        "ranked intraday gap, lagged-return ts-rank, absolute "
        "VWAP-ADV20 correlation, and long-term MA deviation × gap."
    )
    window_length = 230
    inputs = ["open", "close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        close = data["close"]
        volume = data["volume"]
        ret = returns(data)
        adv20 = adv(volume, 20)
        v = vwap(data)
        gap = close - open_

        t1 = 2.21 * rank(correlation(gap, delay(volume, 1), 15))
        t2 = 0.7 * rank(open_ - close)
        t3 = 0.73 * rank(ts_rank(delay(-1.0 * ret, 6), 5))
        t4 = rank(abs_(correlation(v, adv20, 6)))
        t5 = 0.6 * rank((ts_mean(close, 200) - open_) * gap)
        return t1 + t2 + t3 + t4 + t5
