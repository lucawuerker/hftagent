"""Market participants often misjudge the implications of order flow on price movements, leading to inefficiencies. By analyzing the volume of signed traded volume in conjunction with the limit-order-book imbalance, this factor aims to capture the relationship between inventory levels and market sentiment, as firms optimize their order executions based on perceived future value, similar to concepts discussed in the value-based inventory management literature."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import (rank, delta, ts_sum, ts_mean, ts_argmax, ts_argmin, sign, abs_, vwap)

@register_factor
class ValueBasedInventoryFlow(BaseFactor):
    factor_id = "value_based_inventory_flow"
    name = "Value-Based Inventory Flow"
    category = "microstructure"
    inputs = ["trade", "volume", "lobImb", "effLobImb"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        trade = data["trade"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        lob_imbalance = data["lobImb"].fillna(0.0)
        eff_lob_imbalance = data["effLobImb"].fillna(0.0)

        # Calculate the net signed traded volume adjusted for effective LOB imbalance
        adjusted_volume = trade * eff_lob_imbalance

        # Normalize by total volume to get the final signal
        signal = adjusted_volume / volume.replace(0, np.nan)

        # Handle NaN values defensively
        signal = signal.fillna(0.0)

        return signal