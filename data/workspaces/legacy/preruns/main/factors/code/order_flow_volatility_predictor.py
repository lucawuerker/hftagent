"""Studies have shown that significant changes in order flow can lead to changes in volatility, especially in periods of high market activity. As traders react to order imbalances and liquidity changes, the resulting volatility can influence future price movements. This factor aims to capture that predictive relationship by analyzing the order flow dynamics in relation to historical volatility trends."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (correlation, ts_mean, ts_sum, abs_,)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class OrderFlowVolatilityPredictor(BaseFactor):
    """Predicts future volatility based on order flow dynamics."""
    factor_id = "order_flow_volatility_predictor"
    category = "volatility"
    inputs = ["orderFlow", "trdLiq"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        trd_liq = data["trdLiq"].replace(0, np.nan)
        historical_volatility = ts_mean(trd_liq, 60).fillna(0.0)
        correlation_signal = correlation(order_flow, historical_volatility, 60)
        return correlation_signal