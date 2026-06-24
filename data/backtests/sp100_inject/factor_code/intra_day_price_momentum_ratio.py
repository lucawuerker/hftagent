"""Intra-day price momentum can provide insights into short-term market sentiment and liquidity dynamics, where rapid price movements often reflect trader behavior and market reactions to news. This factor capitalizes on the idea that stocks exhibiting strong price momentum within the trading day are likely to continue that trend, influenced by behavioral biases such as herding and overreaction to information. This signal computes the ratio of the current price change over a defined intra-day interval (e.g., the first half of the trading day) relative to the price change over the entire day, capturing the strength of price movement relative to overall volatility. This is derived using the 'close' prices for the specified intervals."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, ts_sum, ts_mean, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntraDayPriceMomentumRatio(BaseFactor):
    factor_id = "intra_day_price_momentum_ratio"
    name = "Intra-Day Price Momentum Ratio"
    category = "momentum"
    description = "Computes the ratio of current price change over a defined intra-day interval relative to the price change over the entire day."
    inputs = ["close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        # Calculate the price change over the first half of the day (assuming 5 bars)
        price_change_half_day = delta(close, 5)
        # Calculate the total price change over the entire day (assuming 10 bars)
        total_price_change = delta(close, 10)
        # Calculate the ratio of the two changes
        momentum_ratio = price_change_half_day / total_price_change.replace(0, np.nan)
        # Handle NaN values by filling them with 0.0
        return momentum_ratio.fillna(0.0)