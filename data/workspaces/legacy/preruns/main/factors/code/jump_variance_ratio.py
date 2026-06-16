"""Volatility in financial markets is often exacerbated by sudden jumps caused by news or events. By isolating and quantifying the jump variation relative to the total variance, traders can gain insights into price dynamics and potential reversals, as significant jumps can indicate shifts in market sentiment or liquidity constraints. This factor can help identify stocks that are either overreacting to news or underestimating volatility risks."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, stddev, delta


@register_factor
class JumpVarianceRatio(BaseFactor):
    factor_id = "jump_variance_ratio"
    name = "Jump Variance Ratio"
    category = "volatility"
    description = "This signal computes the ratio of jump variance to total realized variance for each stock, capturing the proportion of volatility attributed to significant jumps."
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        # Calculate total realized variance
        realized_variance = stddev(close, 10) ** 2

        # Calculate jump variance using a simple delta approach
        jump_variance = ts_sum(delta(close, 1).replace(0, np.nan).fillna(0.0) ** 2, 10)

        # Calculate the jump variance ratio
        jump_variance_ratio = jump_variance / realized_variance.where(realized_variance != 0, np.nan)

        return jump_variance_ratio