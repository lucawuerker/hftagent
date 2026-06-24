"""When price finishes strong but does so on unusually weak volume, the move is more likely to be a transient liquidity vacuum than genuine information-driven demand. Conversely, a flat or weak close on heavy volume often signals latent accumulation/distribution and can precede continuation once the imbalance is digested. This factor is designed to extract that cross-sectional divergence between end-of-bar price displacement and participation, complementing existing pure price-range and absorption-style signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, adv, log, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class CloseVolumeDivergence(BaseFactor):
    """Close-to-open return divided by normalized volume surprise, smoothed short-term."""

    factor_id = "close_volume_divergence"
    name = "Close-Volume Divergence"
    category = "microstructure"
    description = (
        "Measure the signed return from open to close and divide it by a "
        "normalized volume surprise, then smooth over a short rolling window. "
        "High positive values indicate strong closes on light volume relative "
        "to recent activity; low negative values indicate weak closes on heavy volume."
    )
    window_length = 10
    inputs = ["open", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"].replace(0.0, np.nan)
        close = data["close"]
        volume = data["volume"].fillna(0.0)

        close_open_ret = (close / open_) - 1.0
        vol_baseline = adv(volume, 20)
        vol_surprise = (volume / vol_baseline.replace(0.0, np.nan)) - 1.0
        vol_surprise = vol_surprise.fillna(0.0)

        denom = 1.0 + abs_(vol_surprise)
        raw = close_open_ret / denom
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return ts_mean(raw, 5).fillna(0.0)
