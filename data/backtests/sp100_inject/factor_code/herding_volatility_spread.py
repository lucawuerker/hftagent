"""Market participants often exhibit herding behavior, leading to increased volatility during certain periods, particularly around news events. When traders collectively react to information, it can create temporary spikes in volatility, which may be predictable based on past price movements and trading volume. This factor aims to capture the spread between current volatility and historical volatility levels during moments of high trading volume, indicating potential overreactions or underreactions in the market."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (stddev, adv, ts_mean, delta)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class HerdingVolatilitySpread(BaseFactor):
    factor_id = "herding_volatility_spread"
    name = "Herding Volatility Spread Indicator"
    category = "volatility"
    description = ("This signal computes the difference between the current realized volatility and the historical average volatility over a specified look-back period, adjusted for trading volume. A significant positive spread may indicate overexuberance in prices due to herding behavior, while a negative spread may suggest complacency or underreaction.")
    window_length = 60
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        adv20 = adv(volume, 20)
        realized_vol = stddev(close, 20)
        historical_vol = ts_mean(stddev(close, 60), 60)
        spread = realized_vol - historical_vol
        return spread.where(adv20 < volume, 0.0).fillna(0.0)