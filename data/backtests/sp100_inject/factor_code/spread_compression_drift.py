"""The paper on bid-ask spreads documents strong intraday structure and persistence in spread dynamics, with wide spreads tending to cluster around market openings and narrower spreads emerging as the book normalizes. When a bar’s trading range compresses relative to recent history while closing near the bar high, it often signals a temporary improvement in liquidity and price discovery rather than mere stasis. I expect subsequent continuation to be stronger in names where this compression occurs on unusually low realized range and high close location, because it reflects latent order-book tightening without immediate exhaustion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, rank, ts_min, ts_mean, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class SpreadCompressionDrift(BaseFactor):
    """Compression-with-close-high microstructure drift signal."""

    factor_id = "spread_compression_drift"
    name = "Spread Compression Drift"
    category = "microstructure"
    description = (
        "High when the bar range is unusually compressed, the close is near "
        "the high, and volume confirms the setup."
    )
    window_length = 20
    inputs = ["high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        bar_range = (high - low).abs()
        denom = bar_range.replace(0, np.nan)

        close_location = (close - low).divide(denom).clip(lower=0.0, upper=1.0).fillna(0.5)
        compression = (ts_mean(bar_range, 20).replace(0, np.nan).divide(bar_range.replace(0, np.nan))).fillna(0.0)
        compression = compression.clip(lower=0.0)

        low_range_rank = 1.0 - rank(bar_range)
        recent_min_bias = 1.0 - ts_rank(bar_range, 20)
        location_bias = close_location

        vol20 = adv(volume, 20)
        vol_confirm = (volume.divide(vol20.replace(0, np.nan)) - 1.0).clip(lower=-1.0, upper=2.0).fillna(0.0)
        vol_confirm = vol_confirm.where(volume.notna(), 0.0)

        raw = compression * (0.5 * low_range_rank + 0.5 * recent_min_bias) * (0.5 + location_bias)
        raw = raw * (1.0 + 0.25 * vol_confirm.clip(lower=0.0))

        smooth = ts_mean(raw, 3)
        return smooth.fillna(0.0)
