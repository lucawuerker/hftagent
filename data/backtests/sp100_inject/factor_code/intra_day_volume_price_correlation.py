"""High trading volume can indicate strong conviction among market participants. When there is a strong positive correlation between volume and price changes within the same trading day, it suggests that the trend is likely to continue. This behavior aligns with theories of momentum in behavioral finance, where traders tend to follow the crowd during periods of high activity. This factor computes the correlation between intra-day price changes and corresponding trading volume, capturing the strength of price movements driven by volume. It assesses whether higher volume days correspond with larger price changes, indicating potential continuation of the trend."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import correlation, delta, adv
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntraDayVolumePriceCorrelation(BaseFactor):
    factor_id = "intra_day_volume_price_correlation"
    name = "Intra-Day Volume-Price Correlation"
    category = "momentum"
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        price_change = delta(close, 1)
        adv_volume = adv(volume, 20)
        correlation_result = correlation(price_change, volume, 20)
        return correlation_result.where(adv_volume > 0, 0.0)