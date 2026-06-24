"""When a bar’s range expands while the close drifts toward the low and volume rises, it often reflects liquidity being withdrawn faster than price can adjust. This is the same latent deterioration mechanism emphasized in the paper on early detection of microstructure regimes: stress is typically preceded by a build-up phase where adverse conditions intensify before they become obvious in the close-to-close move. Cross-sectionally, names showing the strongest combination of range expansion, weak close location, and rising traded activity should be under short-term pressure relative to peers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank, ts_mean, ts_rank, delta
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntrabarDepthErosionAcceleration(BaseFactor):
    """Intrabar fragility score from range expansion, weak close location, and volume surprise."""

    factor_id = "intrabar_depth_erosion_acceleration"
    name = "Intrabar Depth Erosion Acceleration"
    category = "microstructure"
    description = (
        "Measures an intrabar fragility score from the interaction of range "
        "expansion, close-location weakness, and volume surprise, then ranks "
        "stocks cross-sectionally. Higher values indicate bars where liquidity "
        "appears to be eroding faster than the final close alone would suggest."
    )
    window_length = 20
    inputs = ["high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        rng = (high - low).replace(0.0, np.nan)
        prev_rng = rng.shift(1)
        range_expansion = (rng / prev_rng) - 1.0
        range_expansion = range_expansion.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        close_location = ((close - low) / rng).replace([np.inf, -np.inf], np.nan).fillna(0.5)
        close_weakness = 1.0 - close_location

        vol_avg = ts_mean(volume, 20)
        vol_surprise = (volume / vol_avg) - 1.0
        vol_surprise = vol_surprise.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        vol_accel = delta(ts_rank(volume, 20), 1).fillna(0.0)

        fragility_raw = (
            ts_rank(range_expansion, 20)
            * ts_rank(close_weakness, 20)
            * ts_rank(vol_surprise, 20)
        )
        fragility_raw = fragility_raw + 0.25 * rank(vol_accel)
        fragility_raw = fragility_raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return rank(fragility_raw)
