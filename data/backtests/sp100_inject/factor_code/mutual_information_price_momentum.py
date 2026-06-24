"""Building on the insights of Fiedor (2014), this factor captures the price momentum by analyzing the mutual information between the current price and its lagged values. Stocks that show a higher mutual information score with their previous prices are likely to continue their momentum, as they indicate a stronger predictive relationship, suggesting that recent price trends could persist longer than expected."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, ts_sum, log, abs_, correlation
from quant_fund_agent.factors.registry import register_factor


@register_factor
class MutualInformationPriceMomentum(BaseFactor):
    factor_id = "mutual_information_price_momentum"
    name = "Mutual Information Price Momentum"
    category = "momentum"
    description = ("The factor is computed by calculating the mutual information between the current price and its previous prices over a specified lag period. A higher mutual information value suggests a stronger momentum signal, indicating the potential for continued price movement in the same direction.")
    inputs = ["close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        lagged_close = close.shift(1).fillna(0.0)
        mutual_info = correlation(close, lagged_close, 5)  # Using a 5-bar window for correlation
        return mutual_info