"""Volume spikes often indicate a temporary imbalance in supply and demand, leading to subsequent price reversals. When a stock experiences a significant increase in volume without a corresponding price move, it suggests that the market may be overreacting, creating an opportunity for mean reversion. This aligns with behavioral finance theories that highlight the tendency of prices to revert to their mean following such imbalances."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import adv, ts_mean, delta, sign


@register_factor
class IntradayVolumeSpikeReversal(BaseFactor):
    factor_id = "intraday_volume_spike_reversal"
    category = "mean_reversion"
    inputs = ["volume", "close"]
    
    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"].fillna(0.0)
        close = data["close"]
        adv5 = adv(volume, 5)
        volume_ratio = volume / adv5
        price_change = delta(close, 1)
        signal = sign(price_change) * volume_ratio
        return signal.where(adv5 > 0, 0.0)