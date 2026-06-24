"""The paper motivates a weakly consistent way to estimate time-varying correlation by aggregating pairwise interactions across the full history with distance-based weights. In markets, a stock whose intrabar range behavior becomes less synchronized with the cross-sectional range of its peers is often in a temporary dislocation regime: local news or order-flow shocks push its path out of phase with the broader sector, and that gap tends to partially normalize once the shock is digested. This factor looks for names whose recent return-path geometry has diverged from their own longer-horizon correlation structure, favoring mean reversion when the divergence is extreme and momentum when it is reinforcing a broad co-move regime."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import correlation, delta, rank, stddev
from quant_fund_agent.factors.registry import register_factor


@register_factor
class DynamicRangeCovarianceMisalignment(BaseFactor):
    """Dynamic short-vs-long correlation misalignment scaled by volatility."""

    factor_id = "dynamic_range_covariance_misalignment"
    name = "Dynamic Range Covariance Misalignment"
    category = "statistical_arbitrage"
    description = (
        "Compare short-horizon and long-horizon dynamic correlation of each "
        "name versus the equal-weight basket, then scale the misalignment by "
        "realized volatility."
    )
    window_length = 60
    inputs = ["close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        rets = close.pct_change()

        basket = rets.mean(axis=1)
        basket_df = pd.DataFrame(
            np.repeat(basket.to_numpy()[:, None], close.shape[1], axis=1),
            index=close.index,
            columns=close.columns,
        )

        short_corr = correlation(rets, basket_df, 10)
        long_corr = correlation(rets, basket_df, 60)
        misalignment = short_corr - long_corr

        vol = stddev(rets, 20).replace(0.0, np.nan)
        raw = misalignment / vol

        # Strengthen the regime interpretation with recent return direction.
        mom = delta(close, 5) / close.shift(5)
        mom = mom.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        regime = rank(mom)

        signal = raw.mul(2.0 * regime - 1.0)
        signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal
