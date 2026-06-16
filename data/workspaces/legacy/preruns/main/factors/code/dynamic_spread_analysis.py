"""The bid-ask spread often reflects the liquidity and market sentiment around a stock. A sudden increase in spread may indicate an impending price movement driven by increased uncertainty or market events. By analyzing the spread dynamics over time, traders can identify potential reversals, as periods of high spread may precede a return to lower spread levels, suggesting a mean-reversion opportunity."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_mean, stddev


@register_factor
class DynamicSpreadAnalysis(BaseFactor):
    factor_id = "dynamic_spread_analysis"
    name = "Dynamic Spread Analysis"
    category = "microstructure"
    inputs = ["spread"]
    window_length = 60

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        spread = data["spread"].fillna(0.0)
        mean_spread = ts_mean(spread, 60)
        std_spread = stddev(spread, 60)
        z_score = (spread - mean_spread) / std_spread
        return z_score.where(std_spread != 0, 0.0)