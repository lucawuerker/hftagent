"""Alpha#38: (-1 * rank(Ts_Rank(close, 10))) * rank(close / open)"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha038(BaseFactor):
    factor_id = "alpha_038"
    name = "Alpha#38"
    category = "momentum"
    description = (
        "Negative ranked 10-day time-series rank of close multiplied "
        "by ranked intraday return (close/open).  Fades stocks that "
        "are both at local highs and had a strong intraday move."
    )
    window_length = 10
    inputs = ["open", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        open_ = data["open"]
        a = -1.0 * rank(ts_rank(close, 10))
        b = rank(close / open_.replace(0, float("nan")))
        return a * b
