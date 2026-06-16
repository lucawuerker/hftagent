"""The liquidity spread between the bid-ask prices can highlight market inefficiencies. When liquidity is low, the spread tends to widen, indicating potential mispricing that can be exploited through mean-reversion strategies. This aligns with the idea that markets tend to return to a normalized spread when liquidity conditions improve."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_mean, ts_sum, ts_rank


@register_factor
class LiquiditySpreadIndicator(BaseFactor):
    factor_id = "liquidity_spread_indicator"
    name = "Liquidity Spread Indicator"
    category = "microstructure"
    description = ("This factor computes the ratio of the bid-ask spread to the average top-of-book depth, "
                   "capturing relative liquidity conditions. A higher value suggests a wider spread "
                   "relative to available liquidity, indicating potential trading opportunities.")
    inputs = ["spread", "depth"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        spread = data["spread"].fillna(0.0)
        depth = data["depth"].replace(0, np.nan)

        avg_depth = ts_mean(depth, 5)
        liquidity_spread = spread / avg_depth

        return liquidity_spread.where(avg_depth.notna(), np.nan)