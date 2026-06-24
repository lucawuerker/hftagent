"""High trading volumes often signify strong investor interest and can lead to continued price momentum. This aligns with behavioral finance theories suggesting that high volume can indicate conviction in a price trend, prompting further buying or selling. Analyzing price movements after significant volume spikes can reveal potential continuations in the direction of that momentum."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (rank, delta, adv, ts_rank)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class HighVolumePriceMomentum(BaseFactor):
    factor_id = "high_volume_price_momentum"
    name = "High Volume Price Momentum"
    category = "momentum"
    description = ("This factor calculates the momentum of a stock's price following periods of unusually high trading volume, "
                   "normalized by average past volume. It aims to identify assets that are likely to continue trending based on recent trading activity.")
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        adv20 = adv(volume, 20)
        d5 = delta(close, 5)
        momentum_signal = ts_rank(d5, 60)
        active = momentum_signal.where(volume > adv20, 0.0)
        return active