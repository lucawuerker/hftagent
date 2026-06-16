"""Recent research has indicated that trading volume distributions often exhibit fat tails, suggesting that extreme volume changes are more likely than standard normal distribution would predict. This can lead to momentum effects where stocks experiencing unusually high trading volume are likely to continue moving in the same direction, as these spikes often indicate increased investor interest or sentiment. The persistence of volume as a signal of underlying market dynamics suggests that capturing these extremes can lead to profitable trading opportunities."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_mean, delta, ts_rank, adv
from quant_fund_agent.factors.registry import register_factor


@register_factor
class FatTailVolumeVariation(BaseFactor):
    factor_id = "fat_tail_volume_variation"
    name = "Fat Tail Volume Variation"
    category = "momentum"
    description = "This factor computes the percentage change in volume over a specified period, highlighting stocks with significant volume spikes compared to their historical averages. It captures the extremes in trading volume, identifying stocks that may be in a momentum phase due to increased market activity."
    inputs = ["volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"].fillna(0.0)
        adv_volume = adv(volume, 20)
        volume_change = delta(volume, 1) / adv_volume
        volume_change = volume_change.replace(0, np.nan)
        return ts_rank(volume_change, 60)