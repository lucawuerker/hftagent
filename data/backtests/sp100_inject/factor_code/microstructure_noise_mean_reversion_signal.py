"""High-frequency trading often exhibits mean-reverting behaviors due to microstructure noise, where prices oscillate around a perceived equilibrium. The semi Markov model highlights the presence of spikes in market activity, leading to volatility clustering and predictable price adjustments. This factor leverages the autocorrelation of returns to identify potential reversals after significant price movements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (  
    returns, ts_mean, ts_rank, ts_sum, stddev, sign, delay,  
)  
from quant_fund_agent.factors.registry import register_factor

@register_factor
class MicrostructureNoiseMeanReversionSignal(BaseFactor):
    factor_id = "microstructure_noise_mean_reversion_signal"
    name = "Microstructure Noise Mean Reversion Signal"
    category = "mean_reversion"
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        returns_series = returns(data)

        # Calculate mean and stddev of returns over the last 5 bars
        mean_returns = ts_mean(returns_series, 5)
        stddev_returns = stddev(returns_series, 5)

        # Calculate the z-score of the returns
        z_score = (returns_series - mean_returns) / stddev_returns

        # Identify significant price movements (e.g., greater than 1 stddev)
        significant_movement = z_score.abs() > 1

        # Apply the mean-reversion signal based on significant movements
        mean_reversion_signal = z_score.where(significant_movement, 0.0)

        return mean_reversion_signal