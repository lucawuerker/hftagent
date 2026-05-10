"""Alpha#19: (-1 * sign((close - delay(close, 7)) + delta(close, 7))) * (1 + rank(1 + sum(returns, 250)))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delay, delta, rank, returns, sign, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha019(BaseFactor):
    factor_id = "alpha_019"
    name = "Alpha#19"
    category = "momentum"
    description = (
        "Contrarian 7-day price direction scaled by ranked 250-day "
        "cumulative return.  Fades the weekly move, but more "
        "aggressively for stocks with strong annual momentum."
    )
    window_length = 250
    inputs = ["close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        ret = returns(data)
        direction = sign((close - delay(close, 7)) + delta(close, 7))
        scale = 1.0 + rank(1.0 + ts_sum(ret, 250))
        return -1.0 * direction * scale
