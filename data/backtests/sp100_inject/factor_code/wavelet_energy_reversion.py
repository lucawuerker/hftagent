"""Compute a short-window realized energy measure from close-to-close returns, then compare it to a slower baseline to form an energy shock z-score. High values indicate an abnormally turbulent recent path versus the stock's own recent history, which is then used as a mean-reversion signal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import returns, ts_mean, stddev
from quant_fund_agent.factors.registry import register_factor


@register_factor
class WaveletEnergyReversion(BaseFactor):
    """Fade unusually high short-horizon return energy relative to its own baseline."""

    factor_id = "wavelet_energy_reversion"
    name = "Wavelet Energy Reversion"
    category = "other"
    description = (
        "Compute a short-window realized energy measure from close-to-close returns, "
        "then compare it to a slower baseline to form an energy shock z-score. "
        "High values indicate an abnormally turbulent recent path versus the stock's "
        "own recent history, which is then used as a mean-reversion signal."
    )
    window_length = 60
    inputs = ["close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        ret = returns(data).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Short-horizon realized energy: sum of squared returns over a brief window.
        short_energy = ts_mean(ret * ret, 5)

        # Slower baseline and dispersion of the energy process itself.
        energy_mean = ts_mean(short_energy, 20)
        energy_std = stddev(short_energy, 20).replace(0.0, np.nan)

        z = (short_energy - energy_mean) / energy_std

        # Mean reversion: unusually energetic paths are faded cross-sectionally.
        signal = -z
        signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal.reindex(index=close.index, columns=close.columns)
