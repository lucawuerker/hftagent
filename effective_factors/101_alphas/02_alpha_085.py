"""Alpha#85: rank(corr(high*0.877+close*0.123, adv30, 10)) ^ rank(corr(Ts_Rank((high+low)/2, 4), Ts_Rank(volume, 10), 7))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, correlation, rank, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha085(BaseFactor):
    prediction_horizon = 6
    factor_id = "alpha_085"
    name = "Alpha#85"
    category = "statistical_arbitrage"
    description = (
        "Ranked 10-day high-close-blend vs ADV30 correlation, raised "
        "to the power of ranked midprice-ts-rank vs volume-ts-rank "
        "correlation."
    )
    window_length = 40
    inputs = ["close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]
        adv30 = adv(volume, 30)
        mid = (high + low) / 2.0

        blend = high * 0.876703 + close * (1.0 - 0.876703)
        base = rank(correlation(blend, adv30, 10))
        exp = rank(correlation(ts_rank(mid, 4), ts_rank(volume, 10), 7))
        return base ** exp
