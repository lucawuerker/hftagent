"""Alpha#57: 0 - 1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import decay_linear, rank, ts_argmax, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha057(BaseFactor):
    factor_id = "alpha_057"
    name = "Alpha#57"
    category = "mean_reversion"
    description = (
        "Negative close-VWAP deviation scaled by decay-linear "
        "smoothed rank of the 30-day argmax of close.  Fades "
        "close-VWAP spread, more aggressively near recent highs."
    )
    window_length = 30
    inputs = ["close", "high", "low"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        v = vwap(data)
        denom = decay_linear(rank(ts_argmax(close, 30)), 2).replace(0, float("nan"))
        return -1.0 * (close - v) / denom
