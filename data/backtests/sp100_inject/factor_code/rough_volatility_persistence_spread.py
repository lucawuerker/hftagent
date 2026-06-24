"""Rough-volatility research suggests that not just the level of volatility, but the persistence of volatility shocks contains predictive information about future price dynamics. When a stock’s realized volatility is elevated and also unusually persistent relative to its own recent history, the market often remains in a risk-off or de-risking phase longer than traders expect, while a sharp drop in persistence can mark stabilization and support mean re-accumulation. This factor tries to separate transient volatility spikes from truly rough, sticky volatility regimes using only bar data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import correlation, decay_linear, delta, rank, returns, stddev, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class RoughVolatilityPersistenceSpread(BaseFactor):
    """Short-minus-long persistence spread in realized volatility, scaled by current volatility."""

    factor_id = "rough_volatility_persistence_spread"
    name = "Rough Vol Persistence Spread"
    category = "volatility"
    description = (
        "Compute a rolling realized-volatility proxy from squared close-to-close returns, then estimate "
        "short-horizon and longer-horizon autocorrelation of that volatility series. The factor is the "
        "short-minus-long persistence spread, optionally scaled by current volatility level, so it is "
        "positive when volatility is both high and unusually self-propagating."
    )
    window_length = 80
    inputs = ["close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]

        ret = returns(data).fillna(0.0)
        rv = ret.pow(2)

        # Realized volatility proxy and a smoother volatility state.
        rv_fast = ts_mean(rv, 5)
        rv_slow = ts_mean(rv, 20)
        vol_state = decay_linear(rv_slow, 10)

        # Persistence proxies: short- and long-horizon autocorrelation of squared returns.
        rv_lag1 = delta(rv, 1)
        rv_lag5 = delta(rv, 5)

        short_persistence = correlation(rv, rv_lag1, 10)
        long_persistence = correlation(rv_slow, rv_lag5, 30)

        spread = short_persistence - long_persistence

        # Scale by current volatility level and emphasize elevated regimes.
        vol_rank = rank(rv_fast)
        vol_scale = (1.0 + vol_rank).fillna(1.0)
        signal = spread * vol_scale * (1.0 + vol_state.fillna(0.0))

        return signal.replace([np.inf, -np.inf], np.nan).fillna(0.0).reindex(index=close.index, columns=close.columns)
