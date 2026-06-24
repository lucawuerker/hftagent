"""The mid-price deviation ratio captures the tendency of prices to revert to the mid-point of the bid-ask spread after significant deviations, reflecting market participants' behavior in correcting overreactions or underreactions. The rapid electronic trading environment influences traders to exploit price inefficiencies when the price strays too far from the mid-price, as demonstrated in the literature on mean reversion in order-driven markets."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, ts_sum, ts_mean, delay, sign
from quant_fund_agent.factors.registry import register_factor


@register_factor
class MidPriceDeviationRatio(BaseFactor):
    factor_id = "mid_price_deviation_ratio"
    name = "Mid-Price Deviation Ratio"
    category = "microstructure"
    description = ("This signal computes the ratio of the absolute difference between the current mid-price and the mid-point of the bid-ask spread, normalized by the bid-ask spread itself. A higher ratio indicates a greater deviation from equilibrium, suggesting potential mean-reversion opportunities.")
    inputs = ["close", "high", "low"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        high = data["high"].fillna(0.0)
        low = data["low"].fillna(0.0)

        mid_price = (high + low) / 2
        deviation = abs_(close - mid_price)
        spread = high - low

        # Avoid division by zero by replacing zeros in spread with NaN
        spread = spread.replace(0, np.nan)
        ratio = deviation / spread

        # Return the ratio, ensuring the same index and columns as close
        return ratio.where(~spread.isna(), 0.0)