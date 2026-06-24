"""Short-horizon realized volatility mean reversion from close-to-close returns.

Short-horizon realized variance tends to overshoot when a name experiences a burst of wide ranges and heavy trading, because transient volatility shocks mean-revert once the information event is digested. Inspired by the paper's emphasis on variance/volatility swap convexity and the distinction between variance level and volatility level, this factor looks for names whose recent realized volatility is elevated relative to their own medium-run baseline. Cross-sectionally, high relative vol names should underperform on a forward-vol basis as volatility normalizes, while low relative vol names should see a smaller drag from mean reversion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import log, rank, returns, ts_mean, stddev
from quant_fund_agent.factors.registry import register_factor


@register_factor
class RealizedVolatilityMeanReversion(BaseFactor):
    """Mean-reversion signal from elevated short-run realized volatility."""

    factor_id = "realized_volatility_mean_reversion"
    name = "Realized Volatility Mean Reversion"
    category = "volatility"
    description = (
        "Computes short-window realized volatility from close-to-close returns, "
        "scales it by a longer-window baseline, and negates the relative spread "
        "so unusually high recent volatility scores as bearish mean reversion."
    )
    window_length = 60
    inputs = ["close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]

        ret = returns(data).replace([np.inf, -np.inf], np.nan)
        ret = ret.fillna(0.0)

        short_window = 10
        long_window = 60

        short_rv = stddev(ret, short_window)
        long_rv = stddev(ret, long_window)

        long_rv_safe = long_rv.replace(0.0, np.nan)
        rv_ratio = short_rv / long_rv_safe
        rv_ratio = rv_ratio.replace([np.inf, -np.inf], np.nan)

        centered_ratio = rv_ratio - ts_mean(rv_ratio, long_window)
        signal = -rank(centered_ratio)

        return signal.reindex(index=close.index, columns=close.columns)
