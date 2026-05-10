"""Alpha#34: rank((1 - rank(stddev(returns,2)/stddev(returns,5))) + (1 - rank(delta(close,1))))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, rank, returns, stddev
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha034(BaseFactor):
    factor_id = "alpha_034"
    name = "Alpha#34"
    category = "volatility"
    description = (
        "Composite of inverted short-to-medium volatility ratio "
        "(2-day vs 5-day return stddev) and inverted ranked 1-day "
        "close change.  Favours stocks where short-term vol is low "
        "relative to medium-term vol and recent price change is small."
    )
    window_length = 5
    inputs = ["close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        ret = returns(data)
        vol_ratio = stddev(ret, 2) / stddev(ret, 5).replace(0, float("nan"))
        a = 1.0 - rank(vol_ratio)
        b = 1.0 - rank(delta(data["close"], 1))
        return rank(a + b)
