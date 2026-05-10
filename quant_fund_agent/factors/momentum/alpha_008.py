"""Alpha#8: -1 * rank((sum(open, 5) * sum(returns, 5)) - delay(sum(open, 5) * sum(returns, 5), 10))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delay, rank, returns, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha008(BaseFactor):
    factor_id = "alpha_008"
    name = "Alpha#8"
    category = "momentum"
    description = (
        "Negative rank of the 10-day change in the product of 5-day "
        "cumulative open and 5-day cumulative returns.  Captures "
        "momentum shocks in the open-return interaction."
    )
    window_length = 15
    inputs = ["open", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        ret = returns(data)
        x = ts_sum(open_, 5) * ts_sum(ret, 5)
        return -1.0 * rank(x - delay(x, 10))
