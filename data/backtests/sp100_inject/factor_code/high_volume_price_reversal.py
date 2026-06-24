"""Research suggests that high trading volume often precedes price reversals, as it indicates strong market interest and can lead to overextension in one direction. When a security experiences a significant price movement alongside high volume, it may create an opportunity for mean reversion as traders take profits or cut losses."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_rank, delta, adv


@register_factor
class HighVolumePriceReversal(BaseFactor):
    factor_id = "high_volume_price_reversal"
    name = "High Volume Price Reversal"
    category = "mean_reversion"
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        adv20 = adv(volume, 20)
        price_change = delta(close, 1)
        high_volume_mask = volume > adv20
        price_reversal_signal = ts_rank(price_change, 60)
        return price_reversal_signal.where(high_volume_mask, 0.0)