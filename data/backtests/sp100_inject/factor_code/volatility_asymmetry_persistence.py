"""Markets often react more violently to downside shocks than to upside moves, and that asymmetry tends to persist for several bars as traders de-risk, widen spreads, or unwind leverage. Inspired by the GJR-GARCH stylized-fact discussion in the provided paper, this factor looks for cases where recent negative returns coincide with expanding intraday range, which can signal continuing risk repricing rather than a one-off move. Cross-sectionally, names with the strongest asymmetric volatility persistence are attractive candidates to underweight or short, while the weakest can be favored on the long side."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class VolatilityAsymmetryPersistence(BaseFactor):
    """Signed persistence of downside-weighted intraday volatility expansion."""

    factor_id = "volatility_asymmetry_persistence"
    name = "Volatility Asymmetry Persistence"
    category = "volatility"
    description = (
        "Compute a signed volatility persistence score from recent bar returns "
        "and range expansion, emphasizing negative-return bars more than positive-return bars."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        prev_close = close.shift(1)
        bar_ret = close.div(open_.replace(0, np.nan)).sub(1.0)
        close_gap = close.div(prev_close.replace(0, np.nan)).sub(1.0)
        intraday_range = high.sub(low).div(open_.replace(0, np.nan))

        downside_flag = (bar_ret < 0.0).astype(float)
        upside_flag = (bar_ret > 0.0).astype(float)

        downside_component = intraday_range.mul(downside_flag).mul(bar_ret.abs())
        upside_component = intraday_range.mul(upside_flag).mul(bar_ret.abs())

        signed_persistence = ts_mean(downside_component, 10).sub(ts_mean(upside_component, 10))
        gap_pressure = ts_mean(close_gap.where(close_gap < 0.0, 0.0).abs(), 5)
        range_acceleration = ts_mean(intraday_range.diff(1).where(intraday_range.diff(1) > 0.0, 0.0), 5)

        raw = signed_persistence.add(0.5 * gap_pressure).add(0.25 * range_acceleration)
        return rank(raw.fillna(0.0))
