"""Alpha#4: -1 * Ts_Rank(rank(low), 9)"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha004(BaseFactor):
    factor_id = "alpha_004"
    name = "Alpha#4"
    category = "mean_reversion"
    description = (
        "Negative time-series rank of cross-sectionally ranked lows "
        "over 9 days.  Bets that stocks at persistent relative lows "
        "will revert upward."
    )
    window_length = 9
    inputs = ["low"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return -1.0 * ts_rank(rank(data["low"]), 9)
