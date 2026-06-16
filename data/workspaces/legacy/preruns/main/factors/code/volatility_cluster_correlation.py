"""Volatility clustering, as observed in the Ising model, suggests that stocks tend to exhibit periods of high or low volatility that can be correlated across different assets. By analyzing the cross-correlation of absolute returns between a stock and others in the S&P 500, we can capture these interdependencies, which may indicate potential price movements based on collective volatility behavior."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import (  
    abs_,  
    correlation,  
    returns,  
    ts_mean,  
)

@register_factor
class VolatilityClusterCorrelation(BaseFactor):
    factor_id = "volatility_cluster_correlation"
    name = "Volatility Cluster Correlation"
    category = "volatility"
    description = "This factor computes the rolling cross-correlation of the absolute returns of a stock with those of all other stocks in the S&P 500 over a specified window. A higher correlation suggests stronger synchronized volatility, which may precede price movements."
    inputs = ["close"]
    window_length = 60

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        abs_returns = abs_(returns(data))
        cross_corr = correlation(abs_returns, abs_returns, self.window_length)
        return cross_corr