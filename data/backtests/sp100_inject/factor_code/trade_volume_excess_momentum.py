"""High-volume trading often leads to stronger momentum in price movements, as it indicates greater conviction by market participants. This aligns with behavioral finance theories, suggesting that increased trading activity can reflect stronger market sentiment and trends. Higher volumes during price increases might indicate that more traders are joining the trend, leading to further price appreciation."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (rank, delta, ts_mean, adv)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class TradeVolumeExcessMomentum(BaseFactor):
    factor_id = "trade_volume_excess_momentum"
    name = "Trade Volume Excess Momentum"
    category = "momentum"
    description = "This factor computes the momentum of price changes adjusted for trade volume. It captures the excess return associated with periods of high trading volume by comparing the price momentum during these periods against periods of lower volume."
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        adv20 = adv(volume, 20)
        momentum = delta(close, 5)
        excess_momentum = momentum / adv20.fillna(1.0)  # Avoid division by zero
        return excess_momentum.where(adv20 > 0, 0.0)  # Mask where adv20 is not positive