"""Anomalies in order flow can indicate significant shifts in market sentiment or impending volatility. By detecting extreme positive or negative deviations in signed order flow (trade) relative to a rolling average, we can identify unusual trading behavior that may precede price movements. This factor leverages the idea that sudden spikes in buyer-initiated or seller-initiated trades can signal a shift in market dynamics."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_mean, ts_max, ts_min, signed_power, stddev, abs_, delay, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class OrderFlowAnomalyDetection(BaseFactor):
    factor_id = "order_flow_anomaly_detection"
    name = "Order Flow Anomaly Detection"
    category = "microstructure"
    inputs = ["trade"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        trade = data["trade"].fillna(0.0)
        mean_trade = ts_mean(trade, 60)
        std_trade = stddev(trade, 60)

        # Calculate z-score
        z_score = (trade - mean_trade) / std_trade.replace(0, np.nan)

        # Masking to avoid NaN issues
        z_score = z_score.where(std_trade != 0, np.nan)

        return z_score