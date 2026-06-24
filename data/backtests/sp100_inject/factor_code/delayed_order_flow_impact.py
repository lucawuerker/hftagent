"""The impact of order flow on prices is often not immediate and can take time to be fully reflected in market prices. By analyzing past volume and price changes, we can gauge the delayed effect of prior trades on current price movements, which may reveal inefficiencies as prices adjust over time. This concept draws from the findings in 'Slow decay of impact in equity markets', highlighting how the effects of trades can linger and influence subsequent price dynamics."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, delta, adv

@register_factor
class DelayedOrderFlowImpact(BaseFactor):
    factor_id = "delayed_order_flow_impact"
    name = "Delayed Order Flow Impact"
    category = "microstructure"
    description = "This factor computes the cumulative impact of order flow over a specified time lag, assessing how past volumes influence current price changes."
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        lagged_volume = adv(volume, 5)
        price_change = delta(close, 1)
        impact = ts_sum(lagged_volume * price_change, 5)
        return impact