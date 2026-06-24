"""Based on the work of Bakosi and Ristorcelli (2013), the Dirichlet distribution can describe the joint behavior of multiple assets as they converge to a stationary distribution. As assets approach their equilibrium states, their returns can show momentum patterns reflective of underlying market dynamics, suggesting that a group of correlated assets will trend together in the same direction. This factor leverages the idea that assets with higher probabilities in the Dirichlet distribution exhibit stronger momentum and can thus provide a trading edge. This signal computes the momentum of asset prices based on their convergence to a Dirichlet distribution, capturing the relative price movements among a set of correlated assets. It uses the historical price data to assess how closely asset returns align with the expected probabilities derived from the Dirichlet distribution."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (rank, delta, ts_sum, ts_mean, stddev, ts_rank, returns)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class DirichletDistributionPriceMomentum(BaseFactor):
    factor_id = "dirichlet_distribution_price_momentum"
    name = "Dirichlet Distribution Price Momentum"
    category = "momentum"
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        momentum = delta(close, 1)
        avg_momentum = ts_mean(momentum, 5)
        avg_volume = ts_mean(volume, 5)
        signal = avg_momentum.where(avg_volume > 0, 0.0)
        return signal