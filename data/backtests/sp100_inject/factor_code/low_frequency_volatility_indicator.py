"""Market volatility can be influenced by the frequency of price movements, particularly at low frequencies. As observed in the paper by Kim and Jun, intraday volatility increases after changes in price limits, indicating that low-frequency price fluctuations may signify underlying market stress or instability. This factor aims to capture such shifts in volatility to identify potential trading opportunities during periods of heightened risk."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, stddev, delay, abs_
from quant_fund_agent.factors.registry import register_factor


@register_factor
class LowFrequencyVolatilityIndicator(BaseFactor):
    factor_id = "low_frequency_volatility_indicator"
    name = "Low Frequency Volatility Indicator"
    category = "volatility"
    description = "This signal computes the average amplitude of low-frequency price movements over a specified intraday period, indicating changes in market volatility."
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        low_freq_amplitude = abs_(delay(close, 5) - close)
        avg_amplitude = ts_sum(low_freq_amplitude, 5) / 5
        volatility_signal = stddev(avg_amplitude, 5)
        return volatility_signal