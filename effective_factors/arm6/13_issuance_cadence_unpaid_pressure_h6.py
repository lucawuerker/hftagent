"""Fresh issuance plus regular untransmitted sell cadence predicts near-term pressure.

Mechanism: a positive disclosed net-stock-issuance innovation identifies a possible
persistent seller.  The signal isolates cases where sell participation is both
regular across recent bars and insufficiently transmitted into price, consistent
with a scheduled program whose remaining inventory is still being distributed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


def _event_age(events: pd.DataFrame) -> pd.DataFrame:
    """Number of causal bars since the latest event, NaN before the first."""
    positions = np.broadcast_to(
        np.arange(len(events), dtype=float)[:, None], events.shape
    )
    stamps = pd.DataFrame(positions, index=events.index, columns=events.columns)
    return stamps - stamps.where(events).ffill()


def _schedule_coherence(pressure: pd.DataFrame, window: int) -> pd.DataFrame:
    """Low dispersion conditional on nonzero pressure is regular execution.

    All moments use a trailing window.  Coherence is separated from intensity so
    that trivially quiet names cannot be classified as scheduled sellers.
    """
    mean_pressure = pressure.rolling(window, min_periods=8).mean()
    dispersion = pressure.rolling(window, min_periods=8).std()
    coefficient_variation = dispersion / mean_pressure.replace(0.0, np.nan)
    return (1.0 - coefficient_variation).clip(lower=0.0, upper=1.0)


@register_factor
class IssuanceCadenceUnpaidPressureH6(BaseFactor):
    factor_id = "issuance_cadence_unpaid_pressure_h6"
    name = "Issuance Cadence Unpaid Pressure"
    category = "microstructure"
    description = (
        "Ranks recent issuer-supply events where regular normalized sell pressure "
        "exceeds its contemporaneous price impact, indicating a likely active "
        "distribution schedule with remaining near-term downside."
    )
    window_length = 60
    inputs = ["high", "low", "close", "volume", "marketCap", "netStockIssuance"]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].astype(float).replace(0.0, np.nan)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        volume = data["volume"].astype(float).clip(lower=0.0)
        market_cap = data["marketCap"].astype(float).where(data["marketCap"] > 0.0)
        issuance = data["netStockIssuance"].astype(float)

        prior_issuance = issuance.shift(1)
        issuance_step = issuance - prior_issuance
        positive_issuance = (
            issuance.notna()
            & prior_issuance.notna()
            & (issuance_step > 0.0)
            & market_cap.notna()
        )
        supply_innovation = (
            issuance_step.where(positive_issuance) / market_cap
        ).clip(lower=0.0, upper=0.25)
        supply_state = supply_innovation.fillna(0.0).ewm(
            halflife=10.0, adjust=False
        ).mean()
        age = _event_age(positive_issuance)

        bar_range = (high - low).abs().replace(0.0, np.nan)
        close_location = ((2.0 * close - high - low) / bar_range).clip(-1.0, 1.0)
        volume_base = volume.rolling(30, min_periods=12).mean().replace(0.0, np.nan)
        relative_volume = (volume / volume_base).clip(lower=0.0, upper=5.0).fillna(0.0)

        sell_pressure = ((-close_location).clip(lower=0.0) * relative_volume).fillna(0.0)
        cadence = _schedule_coherence(sell_pressure, 10)
        recent_intensity = sell_pressure.rolling(10, min_periods=8).mean()
        normal_intensity = sell_pressure.rolling(45, min_periods=15).mean().replace(0.0, np.nan)
        intensity_ratio = (recent_intensity / normal_intensity).clip(lower=0.0, upper=3.0)

        normalized_loss = (
            -(close - close.shift(1)) / bar_range
        ).clip(lower=0.0, upper=1.0).fillna(0.0)
        unpaid_pressure = (sell_pressure - normalized_loss).clip(lower=0.0, upper=4.0)
        unpaid_state = unpaid_pressure.ewm(halflife=3.0, adjust=False).mean()

        valid = (
            close.notna()
            & market_cap.notna()
            & issuance.notna()
            & (age <= 35.0)
        )
        score = (supply_state * cadence * intensity_ratio * unpaid_state).where(valid)
        return rank(score).replace([np.inf, -np.inf], np.nan).fillna(0.5).reindex(
            index=close.index, columns=close.columns
        )
