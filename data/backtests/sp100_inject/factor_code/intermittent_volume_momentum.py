"""High-frequency trading often reveals patterns where intermittent spikes in volume can precede price movements. This can be attributed to market participants reacting to new information or changes in sentiment that manifest as sudden trading bursts. Understanding these volume spikes as momentum indicators can lead to profitable trading opportunities, especially in short time frames."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (rank, delta, adv, ts_mean, ts_rank)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntermittentVolumeMomentum(BaseFactor):
    factor_id = "intermittent_volume_momentum"
    name = "Intermittent Volume Momentum"
    category = "momentum"
    description = ("This factor computes the momentum of a stock's price relative to its volume spikes, identifying periods where volume significantly exceeds its recent average, suggesting potential upcoming price movements. It captures both the magnitude of the volume spike and the subsequent price change over a defined window.")
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        adv20 = adv(volume, 20)
        price_change = delta(close, 5)
        momentum = ts_rank(price_change, 20)
        signal = momentum.where(volume > adv20, 0.0)
        return signal.fillna(0.0)