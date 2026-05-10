"""Alpha#20: (-1 * rank(open - delay(high, 1))) * rank(open - delay(close, 1)) * rank(open - delay(low, 1))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delay, rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha020(BaseFactor):
    factor_id = "alpha_020"
    name = "Alpha#20"
    category = "mean_reversion"
    description = (
        "Three-way interaction of ranked gap components: open vs "
        "prior high, close, and low.  Captures overnight gap "
        "mean-reversion pressure."
    )
    window_length = 2
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        a = -1.0 * rank(open_ - delay(data["high"], 1))
        b = rank(open_ - delay(data["close"], 1))
        c = rank(open_ - delay(data["low"], 1))
        return a * b * c
