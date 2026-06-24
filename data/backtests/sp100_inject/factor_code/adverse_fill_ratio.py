"""Adverse fills occur when limit orders are executed at unfavorable prices due to market movements, leading to immediate losses. By quantifying the ratio of adverse fills to total fills, traders can gain insights into execution quality and the potential for negative price impact, which aligns with the findings in the paper on adverse selection in market-making."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, delta, sign


@register_factor
class AdverseFillRatio(BaseFactor):
    factor_id = "adverse_fill_ratio"
    name = "Adverse Fill Ratio"
    category = "microstructure"
    inputs = ["volume", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"].fillna(0.0)
        close = data["close"].fillna(0.0)

        # Calculate adverse fills: when the price drops after a fill
        adverse_fills = volume.where(delta(close, 1) < 0, 0)
        total_fills = volume

        # Calculate the sum of adverse fills and total fills over a 5-bar window
        adverse_sum = ts_sum(adverse_fills, 5)
        total_sum = ts_sum(total_fills, 5)

        # Calculate the adverse fill ratio, avoiding division by zero
        ratio = adverse_sum / total_sum.replace(0, np.nan)

        return ratio