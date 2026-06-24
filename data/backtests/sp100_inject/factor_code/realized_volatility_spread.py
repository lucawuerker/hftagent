"""High realized volatility in one period followed by a sharp decline can indicate overreaction by the market, leading to a potential mean-reversion opportunity. This aligns with behavioral finance theories where investors may overreact to news, causing temporary spikes in volatility that eventually revert. By capturing this spread between high and low realized volatility periods, traders can exploit these inefficiencies."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_mean, stddev, delay

@register_factor
class RealizedVolatilitySpread(BaseFactor):
    factor_id = "realized_volatility_spread"
    name = "Realized Volatility Spread"
    category = "volatility"
    description = "This factor computes the difference between the realized volatility of a stock over a specified period and the subsequent period, indicating whether the market is overreacting or stabilizing. A high positive spread may suggest a bearish outlook, while a significant negative spread may indicate bullish potential."
    window_length = 60
    inputs = ["close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        realized_volatility_1 = stddev(close, 20)
        realized_volatility_2 = stddev(delay(close, 20), 20)
        spread = realized_volatility_1 - realized_volatility_2
        return spread.fillna(0.0)