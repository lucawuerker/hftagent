"""Markets often absorb small oscillations but reprice sharply when the intrabar path becomes unusually “rough” relative to the realized endpoint move. Inspired by the rough-path perspective in the cited paper, this factor looks for bars where the path complexity is high while net progress is small, which can indicate unstable order-flow and latent jump risk. Cross-sectionally, such names should underperform when the stress is persistent and outperform when the stress is quickly resolved."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import abs_, log, scale, sign, ts_mean


@register_factor
class RoughPathStressSkew(BaseFactor):
    """Normalized intrabar roughness proxy from OHLC path tortuosity."""

    factor_id = "rough_path_stress_skew"
    name = "Rough Path Stress Skew"
    category = "volatility"
    description = (
        "Compute a normalized intrabar roughness proxy from the sum of "
        "absolute open-high, high-low, low-close moves relative to the "
        "absolute close-open return and total range. Higher values indicate "
        "a more tortuous path for the same endpoint move, flagging unstable "
        "price formation."
    )
    window_length = 5
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        oh = abs_(high - open_)
        hl = abs_(high - low)
        lc = abs_(close - low)
        co = abs_(close - open_)
        rng = (high - low).abs()

        path_len = oh + hl + lc
        roughness = path_len / (co + rng + 1e-12)

        # Emphasize persistent roughness relative to local history while keeping
        # the final signal centered and cross-sectionally comparable.
        rough_trend = ts_mean(log(roughness + 1.0), 5)
        endpoint_pressure = sign(close - open_) * log(co + 1.0)
        raw = rough_trend - 0.5 * endpoint_pressure

        signal = scale(raw.fillna(0.0))
        return signal.reindex(index=close.index, columns=close.columns)
