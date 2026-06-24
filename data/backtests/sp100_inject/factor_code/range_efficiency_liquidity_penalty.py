"""Compute intrabar price efficiency as |close-open| divided by true range, then penalize it when the same bar has low volume relative to recent volume or when the candle has large wick-dominated movement. The factor is stronger when efficient price movement occurs on thin participation, and weaker when the same move is broadly absorbed by volume."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, abs_, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class RangeEfficiencyLiquidityPenalty(BaseFactor):
    """Intrabar efficiency penalized by thin volume and wick-heavy candles."""

    factor_id = "range_efficiency_liquidity_penalty"
    name = "Range Efficiency Liquidity Penalty"
    category = "microstructure"
    description = (
        "Compute intrabar price efficiency as |close-open| divided by true range, "
        "then penalize it when the same bar has low volume relative to recent volume "
        "or when the candle has large wick-dominated movement."
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
        true_range = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=0,
        ).groupby(level=0).max()
        true_range = true_range.replace(0.0, np.nan)

        body = (close - open_).abs()
        efficiency = body / true_range

        upper_wick = (high - pd.concat([open_, close], axis=0).groupby(level=0).max()).abs()
        lower_wick = (pd.concat([open_, close], axis=0).groupby(level=0).min() - low).abs()
        wick_total = upper_wick + lower_wick
        wick_ratio = wick_total / true_range

        vol_avg = ts_mean(volume, 20)
        vol_ratio = volume / vol_avg.replace(0.0, np.nan)
        liquidity_penalty = 1.0 / (1.0 + vol_ratio.fillna(0.0))

        wick_penalty = 1.0 / (1.0 + wick_ratio.fillna(0.0))

        raw = efficiency.fillna(0.0) * liquidity_penalty * wick_penalty
        return raw.reindex(index=close.index, columns=close.columns)
