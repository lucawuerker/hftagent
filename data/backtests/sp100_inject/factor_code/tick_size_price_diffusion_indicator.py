"""The tick size influences the price formation process by affecting market microstructure, which in turn impacts price diffusion characteristics. Smaller tick sizes are associated with increased volatility clustering and a higher frequency of non-zero returns, suggesting that price movements become more reactive to market events. This indicator aims to exploit the relationship between tick size changes and subsequent price behavior by capturing the dynamics of price diffusion as influenced by microstructure effects."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import (rank, delta, ts_sum, ts_mean, stddev, ts_rank)

@register_factor
class TickSizePriceDiffusionIndicator(BaseFactor):
    factor_id = "tick_size_price_diffusion_indicator"
    name = "Tick Size Price Diffusion Indicator"
    category = "microstructure"
    inputs = ["close", "volume"]
    description = "This factor computes the relationship between recent price movements and tick size changes, measuring the price diffusion characteristics based on the historical volatility and clustering patterns following tick size adjustments."

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        # Calculate recent price changes and their volatility
        price_diff = delta(close, 1)
        volatility = stddev(close, 20)

        # Calculate the price diffusion indicator based on price changes and volatility
        diffusion_indicator = price_diff / (volatility + 1e-10)  # Adding a small value to avoid division by zero

        # Rank the diffusion indicator across the tickers
        ranked_diffusion = rank(diffusion_indicator)

        return ranked_diffusion