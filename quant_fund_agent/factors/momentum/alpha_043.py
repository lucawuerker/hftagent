"""Alpha#43: ts_rank(volume / adv20, 20) * ts_rank((-1 * delta(close, 7)), 8)"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, delta, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha043(BaseFactor):
    factor_id = "alpha_043"
    name = "Alpha#43"
    category = "momentum"
    description = (
        "Product of 20-day relative-volume ts-rank and 8-day ts-rank "
        "of negative 7-day close change.  Favours high-relative-volume "
        "stocks with recent price declines (contrarian with volume "
        "confirmation)."
    )
    window_length = 27
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"]
        adv20 = adv(volume, 20)
        a = ts_rank(volume / adv20.replace(0, float("nan")), 20)
        b = ts_rank(-1.0 * delta(data["close"], 7), 8)
        return a * b
