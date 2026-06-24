"""Intrabar Trap Reversal Pressure: measures intrabar failure of directional extension by combining signed range excursion with close-to-open recovery and volume confirmation. Higher values indicate bars that pushed strongly away from the open but closed back inside the range, suggesting short-term mean reversion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, rank, sign
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntrabarTrapReversalPressure(BaseFactor):
    """Intrabar failure-of-breakout signal with volume confirmation."""

    factor_id = "intrabar_trap_reversal_pressure"
    name = "Intrabar Trap Reversal Pressure"
    category = "mean_reversion"
    description = (
        "Measures intrabar failure of directional extension by combining signed "
        "range excursion with close-to-open recovery and volume confirmation. "
        "Higher values indicate bars that pushed strongly away from the open but "
        "closed back inside the range, suggesting short-term mean reversion."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)

        bar_range = (high - low).replace(0.0, np.nan)
        upper_excursion = (high - open_).clip(lower=0.0)
        lower_excursion = (open_ - low).clip(lower=0.0)

        direction = sign(close - open_).fillna(0.0)
        excursion = (upper_excursion - lower_excursion) / bar_range
        excursion = excursion.fillna(0.0)

        recovery = (open_ - close) / bar_range
        recovery = recovery.fillna(0.0)
        recovery = recovery.clip(-2.0, 2.0)

        failed_extension = excursion * recovery

        close_location = ((close - low) - (high - close)) / bar_range
        close_location = close_location.fillna(0.0)
        trap_bias = (-1.0 * close_location) * direction

        adv20 = adv(volume, 20).replace(0.0, np.nan)
        vol_ratio = (volume / adv20).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        vol_confirmation = (rank(vol_ratio) - 0.5) * 2.0

        amplitude = ((high - low) / open_.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        amplitude = amplitude.fillna(0.0)
        amplitude = rank(amplitude)

        raw = 0.55 * failed_extension + 0.30 * trap_bias + 0.15 * amplitude
        signal = raw * (1.0 + 0.5 * vol_confirmation)

        signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal
