"""Intraday order-flow often reveals itself through where the bar finishes relative to its full range, but the information content is strongest when that move happens on unusually high participation. The paper on intraday trajectory forecasting emphasizes that volatility and trade activity rise nonlinearly toward critical market times, so a close near the bar high after a volume surge should indicate informed buying pressure, while a close near the low after a volume surge should indicate informed selling pressure. This factor isolates that asymmetry and should complement existing trend and reversal signals by focusing on range settlement under volume confirmation rather than direction alone."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, delta, ts_mean, stddev
from quant_fund_agent.factors.registry import register_factor


@register_factor
class RangeTerminalLiquidityDrift(BaseFactor):
    """Close-location signal weighted by volume surprise."""

    factor_id = "range_terminal_liquidity_drift"
    name = "Range Terminal Liquidity Drift"
    category = "microstructure"
    description = (
        "Compute a signed close-location term, (2*close - high - low)/(high - low), "
        "and weight it by standardized volume surprise using a rolling volume z-score "
        "or volume/rolling-mean volume ratio."
    )
    window_length = 20
    inputs = ["high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        range_den = (high - low).replace(0, np.nan)
        close_location = ((2.0 * close) - high - low) / range_den
        close_location = close_location.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        vol_mean = ts_mean(volume, 20)
        vol_std = stddev(volume, 20)
        vol_z = (volume - vol_mean) / vol_std.replace(0, np.nan)
        vol_z = vol_z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        vol_ratio = volume / vol_mean.replace(0, np.nan)
        vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        volume_surprise = 0.5 * vol_z + 0.5 * (vol_ratio - 1.0)
        volume_surprise = volume_surprise.clip(lower=-5.0, upper=5.0)

        terminal_drift = close_location * volume_surprise
        return terminal_drift.replace([np.inf, -np.inf], np.nan).fillna(0.0)
