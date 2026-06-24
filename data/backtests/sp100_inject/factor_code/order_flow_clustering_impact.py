"""The clustering of buy or sell orders can lead to significant price impact due to the limited availability of liquidity at certain price levels. When multiple orders arrive in quick succession, they can push prices away from their equilibrium, reflecting the urgency of market participants to execute. This aligns with the findings from non-average price impact literature, which suggests that the timing and size of order clusters can vastly influence price trajectories."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (adv, delta, ts_sum, ts_mean, ts_rank, sign)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class OrderFlowClusteringImpact(BaseFactor):
    factor_id = "order_flow_clustering_impact"
    name = "Order Flow Clustering Impact"
    category = "microstructure"
    inputs = ["volume", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"].fillna(0.0)
        close = data["close"]

        # Calculate the rolling sum of volume over the last 5 bars
        volume_sum = ts_sum(volume, 5)

        # Calculate the price change over the last bar
        price_change = delta(close, 1)

        # Calculate the impact of clustered orders as the ratio of price change to volume sum
        impact = price_change / (volume_sum.replace(0, np.nan))

        # Rank the impact values over the last 10 bars
        ranked_impact = ts_rank(impact, 10)

        return ranked_impact