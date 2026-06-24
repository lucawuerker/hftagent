"""Markets often do not absorb information uniformly: some names tend to move first, while related names adjust with a delay. Inspired by the DynLMC paper’s emphasis on lagged inter-channel dependencies, this factor looks for tickers whose current intrabar range expansion is in phase with recent cross-sectional leaders, but whose own close has not yet fully caught up. The edge comes from transient lead-lag spillovers that are too slow to be instantly arbitraged away, especially in clustered sectors and index-linked names."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, correlation, delta, rank, returns, ts_mean, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class LeadLagRangeSync(BaseFactor):
    """Lead-lag synchronization between intrabar range expansion and lagged leader returns."""

    factor_id = "lead_lag_range_sync"
    name = "Lead-Lag Range Synchronization"
    category = "statistical_arbitrage"
    description = (
        "Compute each ticker’s rolling correlation between its current bar range "
        "expansion and the recent lagged returns of the cross-sectional leader basket, "
        "then weight by how far the ticker’s close sits behind its intrabar range midpoint."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        prev_close = close.shift(1)
        intrabar_range = (high - low).divide(prev_close.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        range_expansion = delta(intrabar_range.fillna(0.0), 1)

        ret = returns(data).fillna(0.0)
        leader_score = rank(ret)
        leader_weight = leader_score.where(leader_score >= 0.75, 0.0)
        leader_weight = leader_weight.where(volume.notna(), 0.0)

        leader_basket_ret = ts_sum(ret * leader_weight, 5).divide(
            ts_sum(leader_weight, 5).replace(0.0, np.nan)
        )
        leader_basket_ret = leader_basket_ret.fillna(0.0)
        lagged_leader_ret = leader_basket_ret.shift(1)

        sync = correlation(range_expansion, lagged_leader_ret, 10).fillna(0.0)

        range_mid = (high + low) * 0.5
        close_lag = (range_mid - close).divide((high - low).replace(0.0, np.nan))
        close_lag = close_lag.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        local_congestion = ts_mean(abs_(delta(close, 1)).fillna(0.0), 5).fillna(0.0)
        liquidity_stability = ts_mean(volume.fillna(0.0), 5).divide(
            ts_mean(volume.fillna(0.0), 20).replace(0.0, np.nan)
        )
        liquidity_stability = liquidity_stability.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        signal = sync * close_lag * (1.0 + 0.5 * local_congestion) * (1.0 + 0.25 * liquidity_stability)
        signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal.reindex(index=close.index, columns=close.columns)
