"""The theory of rank reversal suggests that stocks exhibiting extreme changes in their market capitalization ranks are likely to revert to their previous ranks over time. This behavior is driven by investor sentiment and market inefficiencies that create temporary overreactions to news or earnings events, leading to strong price momentum in the direction opposite to the rank change. The study by Li and Papanicolaou emphasizes the enhanced mean-reverting properties of residual returns in rank space, providing a potential edge in exploiting these dynamics.

This factor computes the momentum of stocks that have recently experienced significant upward or downward shifts in their market capitalization rank, capturing the tendency for these stocks to revert in price towards their previous rank position."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (rank, delta, ts_rank, ts_mean, adv)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class RankReversalMomentum(BaseFactor):
    factor_id = "rank_reversal_momentum"
    name = "Rank Reversal Momentum"
    category = "momentum"
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        adv20 = adv(volume, 20)
        rank_change = delta(rank(close), 1)  # 1-bar rank change
        momentum = ts_rank(close.pct_change(), 60)  # 60-bar momentum

        # Create a mask for significant rank changes
        significant_change = abs(rank_change) > 0.1  # Threshold for significant change

        # Calculate the final output
        output = momentum.where(significant_change, 0.0)

        return output.fillna(0.0)