"""Markets often experience price movements driven by hidden liquidity, which can signal the intention of informed traders. By measuring the sensitivity of price changes to hidden volume fluctuations, we can identify stocks that are likely to respond sharply to underlying demand pressures, thus capturing potential short-term price reversals or continuations."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import correlation, delta, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class HiddenVolumeSensitivity(BaseFactor):
    factor_id = "hidden_volume_sensitivity"
    name = "Hidden Volume Sensitivity"
    category = "microstructure"
    inputs = ["close", "hidden"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        hidden = data["hidden"].fillna(0.0)

        # Calculate price changes
        price_change = delta(close, 1)

        # Calculate rolling correlation between price changes and hidden volume
        sensitivity = correlation(price_change, hidden, 60)

        # Handle NaN values defensively
        return sensitivity.replace(0, np.nan)