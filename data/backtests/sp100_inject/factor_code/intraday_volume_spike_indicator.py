"""High intraday volume spikes often indicate significant market interest and can precede price movements. According to the findings from the IVE paper, the relationship between predicted volume standard deviation and actual volume ratios suggests that spikes in volume can signal increased volatility and potential price changes, offering traders an edge in timing their trades."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, delta, ts_mean, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntradayVolumeSpikeIndicator(BaseFactor):
    factor_id = "intraday_volume_spike_indicator"
    name = "Intraday Volume Spike Indicator"
    category = "volatility"
    description = "Computes the percentage change in intraday volume compared to a moving average of volume over a defined window."
    inputs = ["volume"]
    window_length = 20

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"].fillna(0.0)
        adv_volume = adv(volume, self.window_length)
        volume_change = delta(volume, 1) / adv_volume
        return ts_rank(volume_change, self.window_length).where(adv_volume > 0, 0.0)