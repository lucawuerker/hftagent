"""Limit order imbalance can serve as a powerful indicator of future price movements. When there is a significant imbalance in limit orders (more buy orders than sell orders or vice versa), it can indicate potential upward or downward pressure on prices. This factor adapts to changing market conditions by analyzing the depth and direction of limit orders over time, allowing traders to capture shifts in market sentiment earlier than traditional measures."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (rank, delta, ts_sum, ts_mean, ts_max, ts_min, ts_argmax, ts_argmin, ts_rank, sign, abs_,)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class AdaptiveLimitOrderImbalance(BaseFactor):
    factor_id = "adaptive_limit_order_imbalance"
    name = "Adaptive Limit Order Imbalance"
    category = "microstructure"
    inputs = ["lobImb", "depth", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        lob_imb = data["lobImb"].fillna(0.0)
        depth = data["depth"].replace(0, np.nan)
        volume = data["volume"].fillna(0.0)

        # Calculate the effective limit order imbalance adjusted by depth
        effective_lob_imb = lob_imb * depth
        effective_lob_imb = effective_lob_imb.fillna(0.0)

        # Normalize by volume to get a ratio
        normalized_lob_imb = effective_lob_imb / volume
        normalized_lob_imb = normalized_lob_imb.replace(np.inf, np.nan)

        # Return the normalized imbalance, ensuring no NaNs are present
        return normalized_lob_imb.fillna(0.0)