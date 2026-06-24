"""Markets with multi-scale roughness often alternate between orderly drift and noisy absorption. Inspired by the mixed-fractional/rough-path view in the paper, this factor looks for bars where directional price progress is achieved with unusually low intrabar path entropy and compressed range occupancy, which can indicate a persistent move that is still under-appreciated cross-sectionally. Conversely, when price makes little net progress but expends a lot of intrabar energy, the move is more likely to be mean-reverting or continuation-poor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank, ts_mean, ts_sum, stddev
from quant_fund_agent.factors.registry import register_factor


@register_factor
class FractalEnergyImbalance(BaseFactor):
    """Directional efficiency versus normalized intrabar energy."""

    factor_id = "fractal_energy_imbalance"
    name = "Fractal Energy Imbalance"
    category = "other"
    description = (
        "Compute a cross-sectional score from close-to-open return divided "
        "by a normalized intrabar energy term built from high-low range, "
        "close-open displacement, and volume-scaled activity. Penalize bars "
        "where the path is expensive relative to realized directional progress; "
        "reward bars where directional efficiency is high and stable over a "
        "short rolling window."
    )
    window_length = 12
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)

        eps = 1e-12
        prev_close = close.shift(1)
        prev_close = prev_close.where(prev_close.notna(), open_)
        prev_close = prev_close.replace(0.0, np.nan)

        true_range = (high - low).abs().fillna(0.0)
        body = (close - open_).abs().fillna(0.0)
        net_progress = (close - prev_close).fillna(0.0)
        directional_return = (net_progress / prev_close).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        range_norm = true_range / prev_close.abs().replace(0.0, np.nan)
        range_norm = range_norm.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        body_norm = body / prev_close.abs().replace(0.0, np.nan)
        body_norm = body_norm.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        vol_base = ts_mean(volume.fillna(0.0), 5)
        vol_ratio = volume / vol_base.replace(0.0, np.nan)
        vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        vol_energy = np.log1p(vol_ratio.clip(lower=0.0))

        energy = 0.5 * range_norm + 0.35 * body_norm + 0.15 * vol_energy
        energy = energy.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        efficiency = directional_return / (energy + eps)
        efficiency = efficiency.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        short_eff_mean = ts_mean(efficiency, 3)
        short_eff_std = stddev(efficiency, 3).replace(0.0, np.nan)
        stable_eff = short_eff_mean / short_eff_std
        stable_eff = stable_eff.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        pressure = (range_norm + body_norm + vol_energy).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pressure_z = pressure - ts_mean(pressure, 5)
        pressure_z = pressure_z / (stddev(pressure, 5).replace(0.0, np.nan) + eps)
        pressure_z = pressure_z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        raw = 0.7 * efficiency + 0.2 * stable_eff - 0.3 * pressure_z
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return rank(raw)
