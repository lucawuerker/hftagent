"""Range-State Entropy Momentum: measure a rolling momentum term from close-to-close returns, then multiply it by the negative of a rolling intrabar path entropy proxy built from the relative positions of open, high, low, and close within each bar."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, rank, returns, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class RangeStateEntropyMomentum(BaseFactor):
    """Cross-sectional momentum penalized by intrabar path entropy."""

    factor_id = "range_state_entropy_momentum"
    name = "Range-State Entropy Momentum"
    category = "momentum"
    description = (
        "Measure a rolling momentum term from close-to-close returns, then multiply it by the "
        "negative of a rolling intrabar path entropy proxy built from the relative positions of "
        "open, high, low, and close within each bar. The signal is cross-sectional: names with "
        "strong recent returns and unusually orderly intraday price paths rank long, while strong "
        "returns with noisy, high-entropy paths rank short."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        prev_close = close.shift(1)
        bar_range = (high - low).replace(0.0, np.nan)

        open_pos = ((open_ - low) / bar_range).clip(0.0, 1.0).fillna(0.5)
        high_pos = ((high - low) / bar_range).clip(0.0, 1.0).fillna(1.0)
        low_pos = ((low - low) / bar_range).clip(0.0, 1.0).fillna(0.0)
        close_pos = ((close - low) / bar_range).clip(0.0, 1.0).fillna(0.5)

        step_oh = abs_(open_pos - high_pos)
        step_hl = abs_(high_pos - low_pos)
        step_lc = abs_(low_pos - close_pos)
        step_oc = abs_(open_pos - close_pos)

        path_dispersion = (step_oh + step_hl + step_lc + step_oc) / 4.0
        orderedness = 1.0 - path_dispersion.clip(0.0, 1.0)

        p1 = open_pos.clip(1e-6, 1.0 - 1e-6)
        p2 = high_pos.clip(1e-6, 1.0 - 1e-6)
        p3 = low_pos.clip(1e-6, 1.0 - 1e-6)
        p4 = close_pos.clip(1e-6, 1.0 - 1e-6)

        entropy_proxy = -(
            p1 * np.log(p1)
            + (1.0 - p1) * np.log(1.0 - p1)
            + p2 * np.log(p2)
            + (1.0 - p2) * np.log(1.0 - p2)
            + p3 * np.log(p3)
            + (1.0 - p3) * np.log(1.0 - p3)
            + p4 * np.log(p4)
            + (1.0 - p4) * np.log(1.0 - p4)
        ) / 4.0

        entropy_proxy = entropy_proxy.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        intrabar_entropy = ts_mean(entropy_proxy, 5)
        intrabar_order = ts_mean(orderedness, 5)

        momentum = ts_mean(returns(data), 10)
        momentum_cs = rank(momentum)
        order_cs = rank(intrabar_order - intrabar_entropy)

        signal = momentum_cs * (1.0 - 2.0 * order_cs)
        return signal.astype(float)
