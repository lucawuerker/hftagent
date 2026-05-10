"""Alpha#39: (-1*rank(delta(close,7)*(1-rank(decay_linear(volume/adv20, 9))))) * (1+rank(sum(returns,250)))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (
    adv,
    decay_linear,
    delta,
    rank,
    returns,
    ts_sum,
)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha039(BaseFactor):
    factor_id = "alpha_039"
    name = "Alpha#39"
    category = "momentum"
    description = (
        "Negative ranked 7-day close delta adjusted by "
        "decay-linear-smoothed relative volume, scaled by ranked "
        "250-day cumulative return.  Contrarian short-term signal "
        "amplified by long-term momentum."
    )
    window_length = 250
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        ret = returns(data)
        adv20 = adv(volume, 20)

        vol_adj = 1.0 - rank(decay_linear(volume / adv20.replace(0, float("nan")), 9))
        a = -1.0 * rank(delta(close, 7) * vol_adj)
        b = 1.0 + rank(ts_sum(ret, 250))
        return a * b
