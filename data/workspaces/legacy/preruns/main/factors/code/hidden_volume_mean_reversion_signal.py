"""Hidden volume often indicates the presence of informed trading, which can create temporary price movements. If hidden volume spikes in one direction, it may suggest that the price will revert once this pressure dissipates, leading to potential mean reversion opportunities. This reflects the behavior of traders who may have overreacted to new information, creating temporary inefficiencies."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_mean, stddev, ts_sum


@register_factor
class HiddenVolumeMeanReversionSignal(BaseFactor):
    factor_id = "hidden_volume_mean_reversion_signal"
    name = "Hidden Volume Mean Reversion Signal"
    category = "mean_reversion"
    description = "This signal computes the deviation of hidden volume from its moving average over a specified window, identifying significant spikes that may indicate a potential mean reversion point in price. It is calculated as the difference between the hidden volume and its rolling mean, divided by the standard deviation of hidden volume."
    window_length = 60
    inputs = ["hidden"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        hidden_volume = data["hidden"].fillna(0.0)
        mean_hidden_volume = ts_mean(hidden_volume, self.window_length)
        std_hidden_volume = stddev(hidden_volume, self.window_length)

        # Calculate the mean reversion signal
        mean_reversion_signal = (hidden_volume - mean_hidden_volume) / std_hidden_volume

        # Replace NaN values with 0.0 for safety
        return mean_reversion_signal.fillna(0.0)