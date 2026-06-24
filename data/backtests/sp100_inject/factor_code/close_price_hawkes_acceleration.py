"""The paper’s core message is that a mean-reverting state variable with self-exciting jumps can exhibit clustered shocks and persistent excursions before settling back. In intraday markets, repeated close-to-close advances or declines often reflect a locally self-exciting order-flow regime rather than a smooth trend, so the acceleration of recent closes can identify when price pressure is becoming unstable and prone to overshoot. I would expect strong positive acceleration in the close series to have weaker forward continuation than a plain momentum signal, especially after a sequence of compressed bars where jumps start to cluster."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, delta, log, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class ClosePriceHawkesAcceleration(BaseFactor):
    """Cross-sectional close-price acceleration normalized by recent range."""

    factor_id = "close_price_hawkes_acceleration"
    name = "Close Price Hawkes Acceleration"
    category = "other"
    description = (
        "Compute a cross-sectional acceleration score from the second "
        "difference of log close over a short lookback, normalized by "
        "recent average true range proxy from high-low and open-close movement."
    )
    window_length = 8
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        open_ = data["open"]
        high = data["high"]
        low = data["low"]

        log_close = log(close.replace(0, np.nan))
        accel = delta(log_close, 1) - delta(log_close, 2)

        hl_range = (high - low).abs()
        oc_range = (close - open_).abs()
        atr_proxy = ts_mean(hl_range + oc_range, 5)
        atr_proxy = atr_proxy.replace(0.0, np.nan)

        signal = accel / atr_proxy
        signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal
