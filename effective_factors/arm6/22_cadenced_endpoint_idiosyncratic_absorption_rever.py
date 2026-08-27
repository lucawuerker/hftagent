"""Cadenced endpoint absorption reversal under idiosyncratic liquidity conditions.

Mechanism: a regularly scheduled participation shock whose intrabar and
close-to-close impact both fade is likely being absorbed by replenishing
liquidity.  Endpoint agreement verifies that the pressure is execution-related,
while a market-wide range-synchrony veto excludes common information or broad
liquidity-shock episodes that need not mean-revert.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor

def _cadence(activity: pd.DataFrame, window: int) -> pd.DataFrame:
    """Mean trailing short-lag autocorrelation of abnormal participation."""
    pieces = []
    for lag in (1, 2, 3):
        pieces.append(activity.rolling(window, min_periods=window).corr(activity.shift(lag)))
    return ((pieces[0] + pieces[1] + pieces[2]) / 3.0).clip(lower=0.0, upper=1.0)

@register_factor
class CadencedEndpointIdiosyncraticAbsorptionReversal(BaseFactor):
    factor_id = 'cadenced_endpoint_idiosyncratic_absorption_rever'
    name = 'Cadenced Endpoint Idiosyncratic Absorption Reversal'
    category = 'microstructure'
    description = 'Reverses regular endpoint execution pressure only when both intrabar and participation-adjusted close impact fade, excluding synchronized market-wide range shocks.'
    window_length = 40
    inputs = ['open', 'high', 'low', 'close', 'volume']
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_price = data['open'].astype(float).replace(0.0, np.nan)
        high = data['high'].astype(float).replace(0.0, np.nan)
        low = data['low'].astype(float).replace(0.0, np.nan)
        close = data['close'].astype(float).replace(0.0, np.nan)
        volume = data['volume'].astype(float).fillna(0.0).clip(lower=0.0)
        bar_range = (high - low).where(high - low > 0.0)
        volume_base = volume.shift(1).rolling(20, min_periods=20).mean()
        relative_volume = volume / volume_base.replace(0.0, np.nan)
        activity = np.log1p(relative_volume.clip(lower=0.0)).clip(lower=0.0, upper=4.0)
        cadence = _cadence(activity, 15)
        sustained_activity = activity.rolling(3, min_periods=3).mean().clip(upper=4.0)
        signed_impact = ((close - open_price) / bar_range).clip(-1.0, 1.0)
        persistent_direction = signed_impact.rolling(3, min_periods=3).mean()
        intrabar_efficiency = signed_impact.abs()
        current_intrabar = intrabar_efficiency.rolling(2, min_periods=2).mean()
        prior_intrabar = intrabar_efficiency.shift(2).rolling(5, min_periods=5).mean()
        intrabar_fade = ((prior_intrabar - current_intrabar) / prior_intrabar.replace(0.0, np.nan)).clip(lower=0.0, upper=1.0)
        endpoint = ((2.0 * close - high - low) / bar_range).clip(-1.0, 1.0)
        endpoint_state = endpoint.rolling(3, min_periods=3).mean()
        endpoint_agreement = (persistent_direction * endpoint_state).clip(lower=0.0, upper=1.0)
        close_impact = close.pct_change().abs() / np.sqrt(relative_volume.clip(lower=0.1, upper=20.0))
        prior_close_impact = close_impact.shift(1).rolling(15, min_periods=15).mean()
        close_impact_deficit = ((prior_close_impact - close_impact) / prior_close_impact.replace(0.0, np.nan)).clip(lower=0.0, upper=1.0)
        range_fraction = (bar_range / close).replace([np.inf, -np.inf], np.nan)
        prior_range = range_fraction.shift(1).rolling(20, min_periods=20).mean()
        range_shock = range_fraction / prior_range.replace(0.0, np.nan)
        broad_shock_share = (range_shock > 1.5).astype(float).mean(axis=1)
        idiosyncratic_gate = (1.0 - broad_shock_share).clip(lower=0.0, upper=1.0)
        idiosyncratic_gate_panel = pd.DataFrame(np.repeat(idiosyncratic_gate.to_numpy()[:, None], close.shape[1], axis=1), index=close.index, columns=close.columns)
        score = -persistent_direction * sustained_activity * cadence * intrabar_fade * close_impact_deficit * endpoint_agreement * idiosyncratic_gate_panel
        return rank(score.replace([np.inf, -np.inf], np.nan)).fillna(0.5)