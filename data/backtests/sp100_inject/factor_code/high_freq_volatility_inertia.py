"""The paper argues that the highest-frequency component of realized volatility carries the most information for future volatility, while longer-horizon components contribute less. In liquid markets, short-horizon variance clustering often reflects immediate order-flow turbulence and transitory risk repricing, which tends to persist into the next bar more than directional return signals do. This factor seeks to capture that near-term volatility continuation by comparing very short-horizon realized movement to slower background volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import returns, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class HighFrequencyVolatilityInertia(BaseFactor):
    """Short-horizon realized volatility relative to slower baseline volatility."""

    factor_id = "high_freq_volatility_inertia"
    name = "High-Frequency Volatility Inertia"
    category = "volatility"
    description = (
        "Compute a short-window realized volatility proxy from squared "
        "close-to-close returns over a very recent window, and divide it by "
        "a slower-moving realized volatility estimate over a longer window. "
        "The cross-sectional signal is high when a ticker's short-horizon "
        "volatility is elevated relative to its own medium-term baseline, "
        "indicating persistent turbulence."
    )
    window_length = 20
    inputs = ["close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        rets = returns(data)

        # Short-run realized volatility proxy: recent sum of squared returns.
        short_var = ts_sum(rets.pow(2), 3)

        # Slower baseline realized volatility proxy: longer sum of squared returns.
        long_var = ts_sum(rets.pow(2), 12)

        # Use a small floor to avoid divide-by-zero and preserve numeric output.
        denom = long_var.replace(0, np.nan)
        signal = short_var / denom

        # If the slower baseline is unavailable, fall back to the short-run proxy.
        signal = signal.where(np.isfinite(signal), short_var)

        return signal.reindex(index=close.index, columns=close.columns)
