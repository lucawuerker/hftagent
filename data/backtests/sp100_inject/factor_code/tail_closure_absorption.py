"""When a bar prints a wide range but closes back near its midpoint, it often signals that the initial shock was absorbed rather than confirmed. This is consistent with a rational microstructure view of transient price pressure and with the paper’s broader point that apparent predictability and excess volatility can arise from the path of prices, not just their level. I would expect the strongest edge when the realized range is large relative to the body and the closing location shows mean-reverting absorption, especially after locally elevated intrabar volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, adv, correlation, decay_linear, delta, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class TailClosureAbsorption(BaseFactor):
    """Wide-range bars that close back near midrange, adjusted for recent volatility."""

    factor_id = "tail_closure_absorption"
    name = "Tail Closure Absorption"
    category = "volatility"
    description = (
        "Measures how much of the intrabar range is rejected by the close, "
        "scaled by contemporaneous range size and recent volatility. High values "
        "indicate large excursions that are not sustained into the close, which "
        "can precede short-horizon reversal."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        rng = (high - low).replace(0, np.nan)
        body = abs_(close - (high + low) * 0.5)
        tail_rejection = 1.0 - (body / rng)
        tail_rejection = tail_rejection.clip(lower=0.0, upper=1.0)

        intrabar_vol = (high - low) / close.replace(0, np.nan).abs()
        recent_vol = ts_mean(intrabar_vol, 10)
        vol_accel = delta(recent_vol, 5)

        range_intensity = rng / ts_mean(rng, 20)
        range_intensity = range_intensity.replace([np.inf, -np.inf], np.nan)

        vol_confirmation = correlation(rng, volume, 10)
        vol_confirmation = vol_confirmation.fillna(0.0)

        absorption = tail_rejection * range_intensity
        absorption = absorption * (1.0 + vol_accel.fillna(0.0))
        absorption = absorption * (1.0 + vol_confirmation.clip(lower=-1.0, upper=1.0))

        pressure_decay = decay_linear(abs_(close - (high + low) * 0.5) / rng, 5)
        pressure_decay = pressure_decay.fillna(0.0)

        signal = absorption * (1.0 - pressure_decay)
        signal = signal * (1.0 + (volume / adv(volume, 20)).replace([np.inf, -np.inf], np.nan).fillna(0.0))
        signal = signal.replace([np.inf, -np.inf], np.nan)
        return signal.fillna(0.0)
