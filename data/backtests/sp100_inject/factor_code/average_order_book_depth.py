"""Research indicates that the depth of the order book affects price resilience and impacts the likelihood of limit orders being executed. A deeper order book around the current price suggests a higher likelihood of order execution, which can correlate with subsequent price movements as traders react to increased liquidity. This factor leverages the relationship between order book depth and market dynamics as outlined in the paper about order books as queueing systems."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum

@register_factor
class AverageOrderBookDepth(BaseFactor):
    factor_id = "average_order_book_depth"
    name = "Average Order Book Depth"
    category = "microstructure"
    inputs = ["volume"]  # Assuming volume is the only relevant input for depth calculation

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"].fillna(0.0)  # Fill NaN values with 0.0
        avg_depth = ts_sum(volume, 5) / 5  # Calculate the average depth over the last 5 bars
        return avg_depth