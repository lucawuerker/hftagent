"""Based on the insights from the paper on optimal investment under price impact, large trades can create significant price movements that impact the potential returns of subsequent trades. This factor captures the idea that securities experiencing high price impacts may lead to constrained trading, resulting in suboptimal investment decisions. Therefore, understanding the price impact on traded securities can help identify potential reversals or continuations in price trends.

This signal computes the price impact ratio by comparing the price changes with the volume traded over a specified period, highlighting securities where large trades have caused significant price movements, potentially indicating constrained trading scenarios.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, ts_sum, ts_mean, sign, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class PriceImpactConstrainedTrading(BaseFactor):
    factor_id = "price_impact_constrained_trading"
    name = "Price Impact Constrained Trading Indicator"
    category = "microstructure"
    description = "This signal computes the price impact ratio by comparing the price changes with the volume traded over a specified period, highlighting securities where large trades have caused significant price movements, potentially indicating constrained trading scenarios."
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        price_change = delta(close, 1)
        total_volume = ts_sum(volume, 5)

        price_impact_ratio = price_change / total_volume.replace(0, np.nan)
        price_impact_ratio = price_impact_ratio.where(~price_impact_ratio.isna(), 0.0)

        return price_impact_ratio