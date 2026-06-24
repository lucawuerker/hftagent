"""This factor leverages the observation that stocks with increasing traded volume tend to outperform in the short term, as higher volume indicates greater investor interest and momentum. The cumulative volume momentum ratio captures the relationship between current momentum and cumulative volume trading, suggesting that stocks with both high momentum and high cumulative volume are positioned for continued upward movement."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, delta, ts_rank, adv
from quant_fund_agent.factors.registry import register_factor


@register_factor
class CumulativeVolumeMomentumRatio(BaseFactor):
    factor_id = "cumulative_volume_momentum_ratio"
    name = "Cumulative Volume Momentum Ratio"
    category = "momentum"
    description = ("The factor computes the ratio of the cumulative volume over a specified time period to the price momentum of each stock, providing a signal that indicates potential upward price movements based on volume trends.")
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        adv20 = adv(volume, 20)
        cumulative_volume = ts_sum(volume, 5).fillna(0.0)
        price_momentum = delta(close, 5).fillna(0.0)
        price_momentum = price_momentum.replace(0, np.nan)
        momentum_ratio = cumulative_volume / price_momentum
        momentum_ratio = momentum_ratio.where(adv20 > 0, 0.0)
        return momentum_ratio