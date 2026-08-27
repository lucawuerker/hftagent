"""Forecast-channel margin rotation: cash-backed EPS revisions can outrun revenue revisions.

Mechanism: analysts update revenue and profitability models through separate
channels.  A positive EPS revision relative to the revenue revision indicates a
margin, mix, or operating-leverage reassessment that investors may initially
discount while anchoring on the more salient sales narrative.  The inference is
more credible when accounting earnings have stronger cash conversion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import indneutralize, rank
from quant_fund_agent.factors.registry import register_factor


def _scaled_forecast_revision(
    estimate: pd.DataFrame,
    lag: int,
    scale_window: int,
) -> pd.DataFrame:
    """Causal, robustly scaled forecast update using only current and past data."""
    prior = estimate.shift(lag)
    scale = estimate.abs().rolling(scale_window, min_periods=lag).median()
    update = (estimate - prior) / scale.replace(0.0, np.nan)
    return update.clip(lower=-4.0, upper=4.0)


@register_factor
class ForecastChannelMarginRotation(BaseFactor):
    factor_id = "forecast_channel_margin_rotation"
    name = "Forecast Channel Margin Rotation"
    category = "sentiment"
    description = (
        "Sector-neutral wedge between robust EPS-consensus and revenue-consensus "
        "updates, weighted toward firms with higher income quality."
    )
    window_length = 126
    inputs = ["close", "epsEstimate", "revenueEstimate", "incomeQuality", "sector"]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        template = data["close"].astype(float)
        eps_est = data["epsEstimate"].astype(float)
        revenue_est = data["revenueEstimate"].astype(float)
        income_quality = data["incomeQuality"].astype(float)

        eps_update = _scaled_forecast_revision(eps_est, 21, 126)
        revenue_update = _scaled_forecast_revision(revenue_est, 21, 126)

        # Percentile transforms make the two economically different forecast
        # channels comparable without assuming identical update variances.
        margin_wedge = rank(eps_update) - rank(revenue_update)
        quality_confidence = rank(income_quality).fillna(0.5)
        raw_signal = margin_wedge * quality_confidence

        sector_neutral = indneutralize(raw_signal, data["sector"])
        return rank(sector_neutral).reindex_like(template).fillna(0.0)
