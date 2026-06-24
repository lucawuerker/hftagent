"""The relationship between volume and price movements can provide insights into market sentiment and the strength of trends. A high volume accompanying a price increase suggests strong buying interest, while a high volume with a price decrease indicates selling pressure, which can lead to potential reversals or continuation patterns. This aligns with the findings in the literature that highlight the importance of volume as a signal of market liquidity and conviction."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, delta, adv


@register_factor
class IntradayVolumePriceRatio(BaseFactor):
    factor_id = "intraday_volume_price_ratio"
    name = "Intraday Volume-Price Ratio"
    category = "microstructure"
    inputs = ["close", "volume"]
    description = (
        "This factor computes the ratio of traded volume to price changes within a specific intraday timeframe, "
        "highlighting stocks that exhibit strong price movements relative to their traded volume. A ratio above "
        "a certain threshold may indicate a significant trend, while lower ratios may suggest weaker price action."
    )

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        adv_volume = adv(volume, 5)
        price_change = delta(close, 1)
        price_change = price_change.replace(0, np.nan)  # Avoid division by zero
        ratio = volume / price_change
        ratio = ratio.where(adv_volume > 0, np.nan)  # Mask where ADV is not positive
        return ratio