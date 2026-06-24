"""Market conditions often shift between different regimes (e.g., bull and bear markets), which can significantly affect asset price movements. By identifying and leveraging price momentum within these regimes, traders can capitalize on sustained price trends that arise during periods of market stability or volatility."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, ts_mean, ts_rank, adv
from quant_fund_agent.factors.registry import register_factor


@register_factor
class RegimeSwitchingPriceMomentum(BaseFactor):
    factor_id = "regime_switching_price_momentum"
    name = "Regime Switching Price Momentum"
    category = "momentum"
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        adv20 = adv(volume, 20)
        momentum = delta(close, 5)
        mean_momentum = ts_mean(momentum, 60)
        active = ts_rank(mean_momentum, 60)
        fallback = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        return active.where(adv20 > volume, fallback)