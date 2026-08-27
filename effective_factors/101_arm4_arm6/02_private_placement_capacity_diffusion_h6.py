"""Private-placement capacity diffusion after a disclosed equity issuance.

Mechanism: a positive issuance accompanied by a disclosed increase in cash can be a
privately absorbed financing rather than public-market distribution.  If a high-ROIC
issuer holds near its daily high while public turnover is muted, the institutional
validation embedded in the placement can diffuse gradually into public prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


def _fresh_update(frame: pd.DataFrame) -> pd.DataFrame:
    """Identify non-initial point-in-time fundamental updates causally."""
    previous = frame.shift(1)
    return frame.notna() & previous.notna() & frame.ne(previous)


@register_factor
class PrivatePlacementCapacityDiffusionH6(BaseFactor):
    factor_id = "private_placement_capacity_diffusion_h6"
    name = "Private Placement Capacity Diffusion"
    category = "microstructure"
    description = (
        "Longs fresh positive issuance disclosures with cash proceeds, strong ROIC, "
        "and near-high prices on muted public turnover, treating them as privately "
        "absorbed financing whose validation diffuses over the following week."
    )
    window_length = 25
    inputs = ["close", "high", "low", "volume", "netStockIssuance", "marketCap", "cash", "roic"]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].where(data["close"] > 0.0)
        high = data["high"]
        low = data["low"]
        volume = data["volume"].where(data["volume"] > 0.0)
        issuance = data["netStockIssuance"]
        market_cap = data["marketCap"].where(data["marketCap"] > 0.0)
        cash = data["cash"]
        roic = data["roic"]

        fresh_issuance = _fresh_update(issuance)
        fresh_cash = _fresh_update(cash)
        issuance_rate = (issuance / market_cap).replace([np.inf, -np.inf], np.nan)
        cash_change_rate = (cash - cash.shift(1)) / market_cap
        cash_change_rate = cash_change_rate.replace([np.inf, -np.inf], np.nan)

        positive_issuance = issuance_rate.clip(lower=0.0)
        cash_proceeds = cash_change_rate.clip(lower=0.0)
        issuer_quality = rank(roic.clip(lower=-1.0, upper=1.0)).fillna(0.5)
        supply_scale = rank(positive_issuance).fillna(0.5)
        proceeds_scale = rank(cash_proceeds).fillna(0.5)

        bar_range = (high - low).replace(0.0, np.nan)
        close_location = ((close - low) / bar_range).clip(lower=0.0, upper=1.0)
        normal_volume = volume.rolling(20, min_periods=10).mean().replace(0.0, np.nan)
        public_participation = (volume / normal_volume).clip(lower=0.0, upper=2.0)
        private_absorption = close_location * (1.0 - public_participation.clip(upper=1.0))

        valid_event = fresh_issuance & fresh_cash & (positive_issuance > 0.0) & (cash_proceeds > 0.0)
        event_score = (supply_scale * proceeds_scale * issuer_quality * private_absorption)
        event_score = event_score.where(valid_event, 0.0).fillna(0.0)

        diffusion = event_score.ewm(halflife=1.5, adjust=False, min_periods=1).mean()
        active = valid_event.rolling(6, min_periods=1).max().fillna(0.0) > 0.0
        ready = close_location.notna() & normal_volume.notna() & market_cap.notna()
        signal = rank(diffusion).fillna(0.5) - 0.5
        return signal.where(active & ready, 0.0).replace([np.inf, -np.inf], 0.0).fillna(0.0).reindex_like(data["close"])
