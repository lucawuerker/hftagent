"""Alpha#17: (-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1)) * rank(ts_rank(volume / adv20, 5))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, delta, rank, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha017(BaseFactor):
    factor_id = "alpha_017"
    name = "Alpha#17"
    category = "momentum"
    description = (
        "Three-way interaction: negative ranked 10-day close ts-rank, "
        "ranked 1-day close acceleration, and ranked relative volume "
        "ts-rank.  Captures momentum reversals amplified by unusual "
        "volume."
    )
    window_length = 20
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        a = -1.0 * rank(ts_rank(close, 10))
        b = rank(delta(delta(close, 1), 1))
        c = rank(ts_rank(volume / adv(volume, 20), 5))
        return a * b * c
