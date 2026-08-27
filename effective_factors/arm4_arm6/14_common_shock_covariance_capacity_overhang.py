"""Common-shock covariance overhang under sparse execution capacity.

Mechanism: a downside market shock activates risk-budget deleveraging. Names whose
market covariance contribution has recently increased are likely mechanical sell
candidates, and the pressure is amplified where ordinary turnover capacity is low.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


def _covariance_budget_innovation(
    ret: pd.DataFrame,
    market_ret: pd.Series,
    weights: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> pd.DataFrame:
    """Trailing short-minus-long covariance contribution per cap weight."""
    market_panel = pd.DataFrame(
        np.repeat(market_ret.to_numpy()[:, None], ret.shape[1], axis=1),
        index=ret.index,
        columns=ret.columns,
    )
    short_cov = (
        (ret * market_panel).rolling(short_window, min_periods=short_window).mean()
        - ret.rolling(short_window, min_periods=short_window).mean()
        * market_panel.rolling(short_window, min_periods=short_window).mean()
    )
    long_cov = (
        (ret * market_panel).rolling(long_window, min_periods=long_window).mean()
        - ret.rolling(long_window, min_periods=long_window).mean()
        * market_panel.rolling(long_window, min_periods=long_window).mean()
    )
    safe_weights = weights.where(weights > 0.0)
    return short_cov.div(safe_weights) - long_cov.div(safe_weights)


@register_factor
class CommonShockCovarianceCapacityOverhang(BaseFactor):
    factor_id = "common_shock_covariance_capacity_overhang"
    name = "Common-Shock Covariance Capacity Overhang"
    category = "microstructure"
    description = (
        "Ranks the negative covariance-budget innovation times a low-turnover "
        "capacity score, but only after an unusually negative common-market shock."
    )
    window_length = 80
    inputs = ["close", "volume", "marketCap"]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
        market_ret = ret.mean(axis=1)

        market_cap = data["marketCap"].reindex_like(close).where(
            data["marketCap"].reindex_like(close) > 0.0
        )
        cap_total = market_cap.sum(axis=1).replace(0.0, np.nan)
        weights = market_cap.div(cap_total, axis=0)

        budget_innovation = _covariance_budget_innovation(
            ret, market_ret, weights, 15, 60
        )
        budget_scale = budget_innovation.rolling(60, min_periods=30).std()
        normalized_innovation = budget_innovation.div(
            budget_scale.replace(0.0, np.nan)
        )

        turnover = data["volume"].reindex_like(close).div(market_cap)
        turnover_capacity = turnover.rolling(20, min_periods=10).mean()
        scarcity = rank(-turnover_capacity)

        market_vol = market_ret.rolling(20, min_periods=15).std()
        downside_common_shock = market_ret < (-0.75 * market_vol)

        raw_signal = -normalized_innovation * scarcity
        signal = rank(raw_signal).where(downside_common_shock, 0.0)
        return signal.reindex_like(close).replace([np.inf, -np.inf], np.nan)
