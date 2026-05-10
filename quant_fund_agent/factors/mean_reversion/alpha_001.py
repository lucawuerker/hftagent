"""Alpha#1: rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2), 5)) - 0.5"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank, returns, signed_power, stddev, ts_argmax
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha001(BaseFactor):
    factor_id = "alpha_001"
    name = "Alpha#1"
    category = "mean_reversion"
    description = (
        "Ts_ArgMax of signed-power transformed input: uses trailing "
        "20-day return volatility when returns are negative, close price "
        "otherwise.  Mean-reversion signal betting on extremes reverting."
    )
    window_length = 25
    inputs = ["close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        ret = returns(data)
        x = close.where(ret >= 0, stddev(ret, 20))
        return rank(ts_argmax(signed_power(x, 2.0), 5)) - 0.5
