"""Alpha#83: (rank(delay((high-low)/(sum(close,5)/5), 2)) * rank(rank(volume))) / ((high-low)/(sum(close,5)/5) / (vwap-close))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delay, rank, ts_mean, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha083(BaseFactor):
    factor_id = "alpha_083"
    name = "Alpha#83"
    category = "microstructure"
    description = (
        "Ranked 2-day lagged normalised range times doubly-ranked "
        "volume, divided by the current normalised range over the "
        "VWAP-close spread.  Captures range-volume dynamics relative "
        "to fair-value deviation."
    )
    window_length = 7
    inputs = ["close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        v = vwap(data)

        ma5 = ts_mean(close, 5).replace(0, float("nan"))
        norm_range = (high - low) / ma5
        vwap_spread = (v - close).replace(0, float("nan"))

        num = rank(delay(norm_range, 2)) * rank(rank(data["volume"]))
        den = (norm_range / vwap_spread).replace(0, float("nan"))
        return num / den
