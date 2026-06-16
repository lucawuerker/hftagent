"""Increased volatility often leads to heightened trading activity, which can be captured by the ratio of order flow to volatility. This ratio can signal the strength of a price movement; high order flow relative to volatility suggests strong conviction behind price changes, while low order flow indicates weak trends. This insight aligns with theories of market microstructure that link order flow with price discovery."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, stddev, abs_, delay


@register_factor
class VolatilityOrderFlowRatio(BaseFactor):
    factor_id = "volatility_order_flow_ratio"
    name = "Volatility Order Flow Ratio"
    category = "volatility"
    inputs = ["orderFlow", "volume", "close"]
    description = "This factor computes the ratio of absolute order flow (traded volume) to the realized volatility over a given period, providing a measure of how much trading activity is driving price changes relative to market volatility."

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        close = data["close"]

        # Calculate realized volatility (standard deviation of close prices)
        realized_volatility = stddev(close, 10).replace(0, np.nan)

        # Calculate absolute order flow
        abs_order_flow = abs_(order_flow)

        # Calculate the ratio of absolute order flow to realized volatility
        ratio = ts_sum(abs_order_flow, 10) / realized_volatility

        # Return the ratio, ensuring it has the same index and columns as close
        return ratio.where(~realized_volatility.isna(), np.nan)