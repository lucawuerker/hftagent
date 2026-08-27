"""Covariance-relieved downside exhaustion.

Mechanism: downside-volatility exhaustion produces a rebound only after the
stock's contribution to market risk has stopped increasing.  A still-rising
covariance budget implies that staggered risk-managed deleveraging can
continue despite a temporary cooling in idiosyncratic downside variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


def _downside_exhaustion(close: pd.DataFrame) -> pd.DataFrame:
    """Causal signed measure of downside excitation curvature."""
    safe_close = close.replace([np.inf, -np.inf], np.nan)
    ret = safe_close.pct_change().replace([np.inf, -np.inf], np.nan)
    ret = ret.clip(lower=-0.95, upper=5.0)

    downside_sq = (-ret.clip(upper=0.0)).pow(2).fillna(0.0)
    prior_var = ret.pow(2).rolling(40, min_periods=20).mean().shift(1)
    innovation = downside_sq / prior_var.replace(0.0, np.nan)
    innovation = innovation.replace([np.inf, -np.inf], np.nan).clip(upper=25.0)
    intensity = innovation.fillna(0.0).ewm(halflife=2.0, adjust=False, min_periods=3).mean()

    intensity_mean = intensity.rolling(30, min_periods=15).mean().shift(1)
    intensity_std = intensity.rolling(30, min_periods=15).std().shift(1)
    burst = ((intensity - intensity_mean) / intensity_std.replace(0.0, np.nan))
    burst = burst.clip(lower=0.0, upper=5.0).fillna(0.0)

    curvature = intensity - 2.0 * intensity.shift(1) + intensity.shift(2)
    curvature_scale = curvature.rolling(30, min_periods=15).std().shift(1)
    curvature_z = curvature / curvature_scale.replace(0.0, np.nan)
    curvature_z = curvature_z.replace([np.inf, -np.inf], np.nan).clip(-5.0, 5.0).fillna(0.0)

    return burst * (-curvature_z)


def _covariance_budget_state(
    close: pd.DataFrame,
    market_cap: pd.DataFrame,
) -> pd.DataFrame:
    """Causal covariance-budget innovation standardized by its own history."""
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
    valid = ret.notna().sum(axis=1)
    market_ret = ret.mean(axis=1).where(valid > 0)
    market_panel = pd.DataFrame(
        np.repeat(market_ret.to_numpy()[:, None], ret.shape[1], axis=1),
        index=ret.index,
        columns=ret.columns,
    )

    caps = market_cap.reindex_like(close).ffill()
    caps = caps.where(caps > 0.0)
    weights = caps.div(caps.sum(axis=1).replace(0.0, np.nan), axis=0)

    short_cov = ret.rolling(15, min_periods=15).cov(market_panel)
    long_cov = ret.rolling(60, min_periods=60).cov(market_panel)
    budget_innovation = (short_cov - long_cov) / weights.replace(0.0, np.nan)

    realized_vol = ret.rolling(20, min_periods=20).std()
    scaled = budget_innovation / realized_vol.replace(0.0, np.nan)
    own_mean = scaled.rolling(60, min_periods=30).mean().shift(1)
    own_std = scaled.rolling(60, min_periods=30).std().shift(1)
    state = (scaled - own_mean) / own_std.replace(0.0, np.nan)
    return state.replace([np.inf, -np.inf], np.nan).clip(-5.0, 5.0)


@register_factor
class CovarianceRelievedDownsideExhaustion(BaseFactor):
    """Trade downside exhaustion only after correlated risk pressure eases."""

    factor_id = "covariance_relieved_downside_exhaustion"
    name = "Covariance-Relieved Downside Exhaustion"
    category = "volatility"
    description = (
        "Ranks downside-excitation deceleration conditional on a falling or "
        "low covariance-budget state, while penalizing acceleration under "
        "rising common-risk pressure."
    )
    window_length = 120
    inputs = ["close", "marketCap"]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        exhaustion = _downside_exhaustion(close)
        covariance_state = _covariance_budget_state(close, data["marketCap"])

        relief_gate = 1.0 / (1.0 + np.exp(covariance_state.fillna(0.0)))
        pressure_gate = 1.0 - relief_gate

        positive_exhaustion = exhaustion.clip(lower=0.0)
        continuing_excitation = (-exhaustion).clip(lower=0.0)
        score = positive_exhaustion * relief_gate - continuing_excitation * pressure_gate

        return (2.0 * rank(score) - 1.0).reindex_like(close).fillna(0.0)
