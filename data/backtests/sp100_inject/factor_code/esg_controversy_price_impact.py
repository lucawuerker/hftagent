"""Research indicates that companies facing ESG-related controversies tend to experience negative price reactions due to investor sentiment and reputational damage (Assael et al., 2022). This factor captures the price impact of controversial events, which can lead to significant underperformance in affected stocks, especially in a market increasingly focused on sustainability and ethical investing."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import (  
    delta, ts_sum, ts_mean, ts_rank, returns, vwap
)

@register_factor
class ESGControversyPriceImpact(BaseFactor):
    factor_id = "esg_controversy_price_impact"
    name = "ESG Controversy Price Impact"
    category = "other"
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        returns_close = returns(data)
        avg_volume = ts_mean(volume, 20)
        price_change = delta(close, 5)
        active = ts_rank(price_change, 60)
        return active.where(volume > avg_volume, 0.0)