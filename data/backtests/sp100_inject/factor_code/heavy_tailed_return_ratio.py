"""In financial markets, assets often exhibit heavy-tailed return distributions, indicating a higher probability of extreme price movements than predicted by normal distributions. By analyzing the ratio of returns that exceed a certain threshold relative to overall returns, traders can identify assets likely to experience heightened volatility or significant price changes, which can present trading opportunities."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import returns, ts_sum, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class HeavyTailedReturnRatio(BaseFactor):
    factor_id = "heavy_tailed_return_ratio"
    name = "Heavy-Tailed Return Ratio"
    category = "volatility"
    inputs = ["close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        ret = returns(data)
        threshold = ts_mean(ret, 20)  # Using the mean return as a threshold
        exceedances = ret.where(ret > threshold, 0)
        ratio = ts_sum(exceedances, 20) / ts_sum(ret, 20)
        return ratio.fillna(0.0)