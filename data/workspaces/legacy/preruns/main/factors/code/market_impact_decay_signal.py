"""The impact of trades on prices is known to decay over time, as highlighted by Brokmann et al. (2014). By capturing the decay of the price impact from recent trades, this factor can identify potential mean-reversion opportunities, as prices often revert to their pre-trade levels. This decay of impact could be more pronounced during periods of high volatility or liquidity, making it a valuable signal in those contexts."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, delta, decay_linear, abs_, sign
from quant_fund_agent.factors.registry import register_factor


@register_factor
class MarketImpactDecaySignal(BaseFactor):
    factor_id = "market_impact_decay_signal"
    name = "Market Impact Decay Signal"
    category = "microstructure"
    inputs = ["trade", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        trade = data["trade"].fillna(0.0)
        close = data["close"]
        volume = data["volume"].fillna(0.0)

        # Calculate the price impact as the change in price due to trades
        price_change = delta(close, 1)
        impact = price_change * sign(trade)

        # Calculate the decay of impact over the last 5 bars
        decay_signal = decay_linear(impact, 5)

        # Normalize by the volume to account for trade size
        normalized_decay = decay_signal / (volume.replace(0, np.nan))

        # Fill NaN values with 0 for the final output
        return normalized_decay.fillna(0.0)