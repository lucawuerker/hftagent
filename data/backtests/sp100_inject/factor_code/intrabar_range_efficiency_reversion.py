"""When price traverses a large intraday range but closes near one end, the move often reflects temporary liquidity imbalance rather than durable information. In a diffusion-style microstructure setting like Horst & Kreher, the book evolves through noisy order arrivals and cancellations, so an inefficient excursion within the bar can mean-revert once the transient flow subsides. Conditioning on range size helps distinguish real trend from mechanical overshoot, which should make the signal complement close-location and wick-based reversals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import abs_, delta, ts_mean


@register_factor
class IntrabarRangeEfficiencyReversion(BaseFactor):
    """Intrabar range-efficiency mean-reversion signal."""

    factor_id = "intrabar_range_efficiency_reversion"
    name = "Intrabar Range Efficiency Reversion"
    category = "mean_reversion"
    description = (
        "Measure the sign of the bar using the distance of close from the bar "
        "midpoint, then scale it by realized intrabar range efficiency "
        "(absolute close-to-open move divided by high-low range)."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        bar_range = (high - low).replace(0.0, np.nan)
        midpoint = 0.5 * (high + low)
        close_from_mid = close - midpoint

        # Sign of the bar from where the close lands relative to the midpoint;
        # large ranges with inefficient closes are emphasized by the range term.
        inefficiency = abs_(close - open_) / bar_range
        range_scale = ts_mean(bar_range, 20)
        range_strength = bar_range / range_scale.replace(0.0, np.nan)

        raw_signal = (-1.0 * close_from_mid / bar_range) * inefficiency * range_strength
        raw_signal = raw_signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Mild stabilization: use the short-term change in the signal to avoid
        # overreacting to static bars and preserve a clean numeric output.
        signal = raw_signal - 0.25 * delta(raw_signal, 1).fillna(0.0)
        return signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
