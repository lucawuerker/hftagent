"""Based on the concepts of fractional cointegration and geometric functionals as discussed in the paper, this factor captures the relationship between the boundary length of price excursions and the area of those excursions. A higher ratio of length to area may indicate overextension in price movements, suggesting potential reversals or mean reversion opportunities. This signal computes the ratio of the excursion length (boundary length) to the area of price excursions above a certain threshold over a specified period. By analyzing the relationship between these two geometric functionals, we can identify potential price reversals.

"""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_max, ts_min, delta


@register_factor
class ExcursionLengthToAreaRatio(BaseFactor):
    factor_id = "excursion_length_to_area_ratio"
    name = "Excursion Length to Area Ratio"
    category = "statistical_arbitrage"
    inputs = ["high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"].fillna(0.0)
        low = data["low"].fillna(0.0)
        close = data["close"].fillna(0.0)

        # Calculate the excursion length (boundary length)
        excursion_length = ts_max(high, 5) - ts_min(low, 5)

        # Calculate the area of price excursions
        area = ts_sum(close, 5)  # This is a simple area calculation over the last 5 bars

        # Calculate the ratio of excursion length to area
        ratio = excursion_length / area.replace(0, np.nan)  # Avoid division by zero

        return ratio