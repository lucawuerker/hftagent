"""Hidden volume often reflects informed trading behavior, where traders execute large orders without impacting the market. A sudden surge in hidden volume, particularly when it diverges from the visible volume, suggests increased interest or sentiment around a stock, which can lead to subsequent price movements. This aligns with findings in market microstructure that highlight the significance of non-displayed liquidity."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_mean, delta, ts_rank


@register_factor
class HiddenVolumeSurge(BaseFactor):
    factor_id = "hidden_volume_surge"
    name = "Hidden Volume Surge"
    category = "microstructure"
    description = "This signal computes the difference between the hidden volume and a rolling average of hidden volume over a defined period. A significant increase in hidden volume compared to this average indicates a potential accumulation or distribution of shares, which may precede price changes."
    inputs = ["hidden", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        hidden_volume = data["hidden"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        rolling_avg_hidden = ts_mean(hidden_volume, 60)
        hidden_volume_change = delta(hidden_volume, 1)
        signal = hidden_volume_change - rolling_avg_hidden
        return signal.where(signal > 0, 0.0).fillna(0.0)