"""When a bar’s intrabar range expands relative to recent history but the close remains pinned near the open, it often reflects aggressive two-sided trading that failed to establish directional control. In liquid markets this kind of “fast but unresolved” action tends to resolve by mean reversion once the temporary order-flow shock is digested. The paper’s emphasis on using transition-consistent state information motivates treating this as a distinct state of market microstructure rather than a pure momentum signal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank, ts_mean, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class RangeAccelerationCongestion(BaseFactor):
    """Range expansion with close-to-open congestion, standardized cross-sectionally."""

    factor_id = "range_acceleration_congestion"
    name = "Range Acceleration Congestion"
    category = "microstructure"
    description = (
        "Compute a standardized ratio of current true range to a rolling "
        "average true range, then penalize bars where the close is near the "
        "open relative to that range. The factor is higher when range expands "
        "sharply and the bar finishes with congestion rather than directional "
        "progress."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        true_range = (high - low).abs()
        atr = ts_mean(true_range, 20)
        atr = atr.replace(0, np.nan)

        range_accel = true_range / atr
        range_accel = range_accel.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        open_close_gap = (close - open_).abs()
        congestion = 1.0 - (open_close_gap / true_range.replace(0, np.nan))
        congestion = congestion.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        congestion = congestion.clip(lower=0.0, upper=1.0)

        accel_state = ts_rank(range_accel, 20).fillna(0.0)
        congestion_state = ts_rank(congestion, 20).fillna(0.0)

        raw = accel_state * congestion_state
        out = rank(raw).fillna(0.0)
        return out
