"""Clustered participation inventory-release factor.

Mechanism: repeated abnormal-volume closes at the same end of the daily range
proxy a self-exciting execution episode.  A fresh event after such clustering
is faded because temporary impact and intermediary inventory should unwind over
the following week.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import indneutralize, rank
from quant_fund_agent.factors.registry import register_factor

def _prior_signed_excitation(flow: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """Causal same-sign exponentially decayed event intensity.

    The shift ensures that the intensity represents events strictly preceding
    today's participation shock, rather than mechanically including it.
    """
    positive = flow.clip(lower=0.0).shift(1).ewm(halflife=halflife, adjust=False, min_periods=3).mean()
    negative = (-flow.clip(upper=0.0)).shift(1).ewm(halflife=halflife, adjust=False, min_periods=3).mean()
    return positive.where(flow >= 0.0, negative)

@register_factor
class ClusteredParticipationInventoryRelease(BaseFactor):
    factor_id = 'clustered_participation_inventory_release_j'
    name = 'Clustered Participation Inventory Release'
    category = 'microstructure'
    description = 'Fades a fresh abnormal-volume close-location event when prior events of the same sign form a causal, exponentially decayed participation cluster.'
    window_length = 25
    inputs = ['close', 'high', 'low', 'volume', 'sector']
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        bar_range = (high - low).replace(0.0, np.nan)
        close_location = ((2.0 * close - high - low) / bar_range).clip(-1.0, 1.0)
        safe_volume = volume.where(volume > 0.0)
        log_volume = np.log(safe_volume)
        volume_mean = log_volume.rolling(21, min_periods=10).mean()
        volume_scale = log_volume.rolling(19, min_periods=10).std().replace(0.0, np.nan)
        volume_surprise = ((log_volume - volume_mean) / volume_scale).clip(-4.0, 4.0)
        participation = volume_surprise.clip(lower=0.0)
        signed_flow = (participation * close_location).replace([np.inf, -np.inf], np.nan)
        prior_intensity = _prior_signed_excitation(signed_flow, 5.0)
        raw_signal = (-signed_flow * prior_intensity).replace([np.inf, -np.inf], np.nan)
        sector = data['sector'].fillna('Unknown')
        sector_relative = indneutralize(raw_signal.fillna(0.0), sector)
        output = (rank(sector_relative).fillna(0.5) - 0.5) * 2.0
        return output.reindex_like(close)