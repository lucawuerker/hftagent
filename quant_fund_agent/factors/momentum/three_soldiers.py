"""Three Advancing White Soldiers – bullish candlestick pattern factor."""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor


@register_factor
class ThreeSoldiersSignal(BaseFactor):
    factor_id = "three_soldiers"
    name = "Three Advancing White Soldiers"
    category = "momentum"
    description = (
        "Detects the bullish three-white-soldiers candlestick pattern: "
        "three consecutive days where each candle opens within the prior "
        "body and closes at a new high."
    )
    window_length = 4
    inputs = ["open", "close"]

    def _check_pattern(self, opens: pd.Series, closes: pd.Series) -> int:
        """Return 1 if the last three bars form the pattern, else 0."""
        if len(opens) < 3 or len(closes) < 3:
            return 0

        o, c = opens.values, closes.values

        # Three consecutive bullish candles
        bullish = all(c[i] > o[i] for i in range(-3, 0))

        # Each close higher than the previous close
        rising_closes = c[-1] > c[-2] > c[-3]

        # Each open within the body of the prior candle
        open_in_body = (
            o[-2] > o[-3] and o[-2] < c[-3] and o[-1] > o[-2] and o[-1] < c[-2]
        )

        return 1 if (bullish and rising_closes and open_in_body) else 0

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        closes = data["close"]
        opens = data["open"]
        signal = pd.DataFrame(0, index=closes.index, columns=closes.columns)

        for ticker in closes.columns:
            for i in range(self.window_length, len(closes)):
                slc = slice(i - self.window_length, i)
                signal.iloc[i, signal.columns.get_loc(ticker)] = (
                    self._check_pattern(opens[ticker].iloc[slc], closes[ticker].iloc[slc])
                )

        return signal
