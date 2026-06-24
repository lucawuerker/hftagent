"""The presence of stochastic volatility can signal potential price movements as it reflects market uncertainty and the changing dynamics of risk perception among traders. Incorporating a measure of instantaneous volatility derived from price movements can exploit the tendency for assets to revert to their mean volatility over time, especially in turbulent market phases, as noted in the paper on option pricing with stochastic volatility."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import stddev, returns
from quant_fund_agent.factors.registry import register_factor


@register_factor
class StochasticVolatilityIndicator(BaseFactor):
    """Computes the stochastic volatility indicator based on price changes."""
    factor_id = "stochastic_volatility_indicator"
    name = "Stochastic Volatility Indicator"
    category = "volatility"
    inputs = ["close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        price_returns = returns(data)
        volatility = stddev(price_returns, 20)
        return volatility