"""During periods of market stress, liquidity tends to evaporate, leading to wider bid-ask spreads and increased transaction costs. By analyzing the ratio of effective volume traded to liquidity available in the order book, we can identify potential mispricings and opportunities when liquidity contracts unexpectedly, as documented in the literature on liquidity stress testing."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, vwap

@register_factor
class LiquidityStressRatio(BaseFactor):
    factor_id = "liquidity_stress_ratio"
    name = "Liquidity Stress Ratio"
    category = "microstructure"
    description = "This signal computes the ratio of effective volume (traded volume) to available liquidity (depth at the best bid and ask) across a set of stocks. It highlights stocks that may exhibit stress in liquidity conditions, providing insights into potential trading opportunities during market downturns."
    inputs = ["trade", "depth", "effSpread"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        trade = data["trade"].fillna(0.0)
        depth = data["depth"].replace(0, np.nan)
        effSpread = data["effSpread"].fillna(0.0)

        # Calculate effective volume traded
        effective_volume = trade

        # Calculate available liquidity
        available_liquidity = depth

        # Calculate liquidity stress ratio
        liquidity_stress_ratio = effective_volume / available_liquidity

        # Handle NaN values
        liquidity_stress_ratio = liquidity_stress_ratio.fillna(0.0)

        return liquidity_stress_ratio