"""Market participants often exhibit behavior influenced by the flow of orders in the market, leading to periodic patterns in price behavior. By analyzing the spectral distance of order flow across stocks, we can identify which stocks are experiencing similar trading behaviors, potentially indicating correlated movements in their prices. This factor is grounded in the idea that similar trading behaviors can lead to similar price movements, as suggested by agent-based models in market dynamics."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, log, abs_, delta, correlation
from quant_fund_agent.factors.registry import register_factor


@register_factor
class SpectralDistanceOrderFlow(BaseFactor):
    factor_id = "spectral_distance_order_flow"
    name = "Spectral Distance of Order Flow"
    category = "microstructure"
    inputs = ["orderFlow"]
    
    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        order_flow = order_flow.replace(0, np.nan)
        
        # Calculate the log of the order flow to use in Kullback-Leibler divergence
        log_order_flow = log(order_flow)
        
        # Calculate the correlation of order flow with a rolling window of 5 bars
        correlation_matrix = correlation(order_flow, order_flow, 5)
        
        # Calculate the spectral distance using Kullback-Leibler divergence
        kl_divergence = ts_sum(log_order_flow, 5) - ts_sum(log_order_flow.where(~order_flow.isna(), 0), 5)
        
        # Normalize the result
        spectral_distance = kl_divergence.fillna(0.0) / (1 + abs_(correlation_matrix))
        
        return spectral_distance