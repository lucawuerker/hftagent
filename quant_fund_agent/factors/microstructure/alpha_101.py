"""Alpha#101: (close - open) / ((high - low) + 0.001)"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha101(BaseFactor):
    factor_id = "alpha_101"
    name = "Alpha#101"
    category = "microstructure"
    description = (
        "Intraday return normalised by the daily range.  Measures "
        "where the close sits relative to the open as a fraction "
        "of the full high-low range."
    )
    window_length = 1
    inputs = ["open", "close", "high", "low"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return (data["close"] - data["open"]) / (data["high"] - data["low"] + 0.001)
