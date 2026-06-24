"""Early in the bar, price often probes beyond the opening range to trigger stops and absorb liquidity, but if that excursion is not sustained by a strong close, it tends to reverse as the transient order-flow imbalance fades. This is a classic microstructure behavior: aggressive liquidity taking creates a short-lived price dislocation that mean-reverts once the sweep is exhausted. The signal should be strongest when the wick is large relative to the full range and volume is elevated, indicating a deliberate sweep rather than random noise."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, abs_, sign
from quant_fund_agent.factors.registry import register_factor


@register_factor
class OpeningRangeLiquiditySweepReversal(BaseFactor):
    """Microstructure signal for opening-range sweep exhaustion and reversal."""

    factor_id = "opening_range_liquidity_sweep_reversal"
    name = "Opening Range Liquidity Sweep Reversal"
    category = "microstructure"
    description = (
        "Measures the degree to which price extends away from the open but fails "
        "to hold that extension by the close, scaled by intrabar range and volume. "
        "Positive values indicate a likely reversal against the sweep direction; "
        "negative values indicate a sustained directional auction."
    )
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        full_range = (high - low).abs().replace(0, np.nan)
        upper_wick = (high - np.maximum(open_, close)).clip(lower=0.0)
        lower_wick = (np.minimum(open_, close) - low).clip(lower=0.0)

        sweep_size = np.maximum(upper_wick, lower_wick)
        sweep_direction = sign((close - open_).where(full_range.notna(), 0.0))
        close_position = ((close - low) / full_range).clip(lower=0.0, upper=1.0)
        reversal_pressure = (0.5 - close_position) * 2.0

        wick_ratio = (sweep_size / full_range).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        volume_rank = adv(volume.fillna(0.0), 20)
        volume_pressure = (volume / volume_rank.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        volume_scale = np.log1p(volume_pressure.clip(lower=0.0))

        sweep_intensity = wick_ratio * volume_scale
        directional_failure = reversal_pressure * (1.0 + abs_(close - open_) / full_range)
        signal = sweep_intensity * directional_failure * (-sweep_direction)

        edge_penalty = ((upper_wick + lower_wick) / full_range).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        signal = signal * (1.0 + edge_penalty)

        return signal.reindex(index=close.index, columns=close.columns)
