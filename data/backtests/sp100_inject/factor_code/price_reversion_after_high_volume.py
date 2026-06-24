"""High trading volume can indicate significant interest or activity in a stock, often leading to price spikes. However, following such spikes, prices tend to revert to their mean as the initial excitement fades and profit-taking occurs. This phenomenon is grounded in behavioral finance, suggesting that traders often overreact to news, leading to temporary mispricings.

This factor computes the average price change over a specified period following bars with volume exceeding a defined threshold, capturing the tendency for prices to revert after high-volume events."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, adv, delta, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class PriceReversionAfterHighVolume(BaseFactor):
    factor_id = "price_reversion_after_high_volume"
    name = "Price Reversion After High Volume"
    category = "mean_reversion"
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        adv20 = adv(volume, 20)
        price_change = delta(close, 5)
        high_volume_mask = volume > adv20
        price_reversion_signal = ts_sum(price_change.where(high_volume_mask, 0), 5)
        return price_reversion_signal