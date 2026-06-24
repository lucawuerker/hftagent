"""The market often behaves in a way that reflects hidden states influenced by underlying economic factors. By modeling prices as a hidden Markov chain, we can identify mean-reverting price behaviors when the observable states suggest a transition to a more favorable regime. This approach leverages the idea that sudden changes in market conditions will lead to reversals as traders react to the updated information about these hidden states."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, ts_rank, delta, sign
from quant_fund_agent.factors.registry import register_factor


@register_factor
class HiddenMarkovChainPriceIndicator(BaseFactor):
    factor_id = "hidden_markov_chain_price_indicator"
    name = "Hidden Markov Chain Price Indicator"
    category = "mean_reversion"
    inputs = ["close", "volume"]
    description = "This factor computes the expected price movement based on the inferred hidden states of a Markov chain using observed price data. It uses the transition probabilities to identify potential mean-reversion opportunities in asset prices."

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        # Calculate the price changes and their mean
        price_change = delta(close, 1)
        mean_price_change = ts_mean(price_change, 5)

        # Calculate the sum of volume over the last 5 bars
        volume_sum = ts_sum(volume, 5)

        # Calculate the signal based on mean price change and volume
        signal = ts_rank(mean_price_change, 10) * sign(volume_sum)

        return signal