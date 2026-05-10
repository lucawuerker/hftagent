"""Alpha#33: rank((-1 * ((1 - (open / close))^1)))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha033(BaseFactor):
    factor_id = "alpha_033"
    name = "Alpha#33"
    category = "mean_reversion"
    description = (
        "Cross-sectional rank of the intraday return (open/close - 1). "
        "Simplifies to rank((open/close) - 1).  Bets on reversion of "
        "the intraday gap."
    )
    window_length = 1
    inputs = ["open", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        close = data["close"]
        intraday = open_ / close.replace(0, float("nan")) - 1.0
        return rank(intraday)
