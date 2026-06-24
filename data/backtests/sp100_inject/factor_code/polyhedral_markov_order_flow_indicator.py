"""The dynamics of order flow can exhibit Markov properties where the influence of past trades impacts future price movements. By leveraging the polyhedral Markov field concept, we can capture the interaction between trades and their impact on price through a structured modeling approach, enhancing the predictive power of order flow signals."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, adv, delta, sign, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class PolyhedralMarkovOrderFlowIndicator(BaseFactor):
    factor_id = "polyhedral_markov_order_flow_indicator"
    category = "microstructure"
    inputs = ["volume", "close"]
    
    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"].fillna(0.0)
        close = data["close"]
        adv20 = adv(volume, 20)
        delta_close = delta(close, 1)
        order_flow_signal = ts_sum(volume, 5) * sign(delta_close)
        active = ts_rank(order_flow_signal, 60)
        return active.where(adv20 > volume, 0.0)