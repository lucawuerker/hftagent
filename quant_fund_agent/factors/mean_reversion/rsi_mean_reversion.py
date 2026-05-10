"""RSI-based mean-reversion factor.

Generates a signal based on the Relative Strength Index crossing
oversold/overbought thresholds, betting on price reversion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor


@register_factor
class RSIMeanReversionSignal(BaseFactor):
    factor_id = "rsi_mean_reversion"
    name = "RSI Mean Reversion"
    category = "mean_reversion"
    description = (
        "Computes the 14-period RSI and maps it to a contrarian signal: "
        "+1 when RSI < oversold threshold (expect bounce), "
        "-1 when RSI > overbought threshold (expect pullback), "
        "0 otherwise."
    )
    window_length = 15  # 14-period RSI + 1 extra bar
    inputs = ["close"]

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> None:
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        closes = data["close"]
        rsi = closes.apply(self._rsi, period=self.period)

        signal = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
        signal[rsi < self.oversold] = 1.0
        signal[rsi > self.overbought] = -1.0
        return signal
