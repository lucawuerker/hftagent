"""Research suggests that queue length in trading can serve as a proxy for market sentiment and pressure, implying that stocks with increasing buying pressure (reflected by queue length) tend to see positive price movements. When volumes are high and queue lengths increase, it indicates heightened interest, potentially leading to upward price momentum as buyers rush to enter positions, thus generating a momentum effect."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import correlation, delta, adv
from quant_fund_agent.factors.registry import register_factor


@register_factor
class QueueLengthPriceMomentum(BaseFactor):
    factor_id = "queue_length_price_momentum"
    name = "Queue Length Price Momentum"
    category = "momentum"
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        adv20 = adv(volume, 20)
        price_momentum = delta(close, 1)
        price_momentum = price_momentum.fillna(0.0)
        correlation_result = correlation(price_momentum, adv20, 5)
        return correlation_result.where(adv20 > 0, 0.0)