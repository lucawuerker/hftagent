"""The paper’s main lesson is that combining coarse and fine spectral views can reveal complementary structure that a single resolution misses. In market data, the same idea applies to price action: if the bar’s open-to-close move is large relative to the bar’s true range and this dominance persists across multiple recent horizons, it often reflects informed order flow rather than random noise. This factor is designed to catch bars where directional pressure is broad-based in time, making continuation more likely than a simple one-bar reversal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor


@register_factor
class WaveletOpenClosePressure(BaseFactor):
    """Multi-horizon open-close pressure with range-normalized directional alignment."""

    factor_id = "wavelet_open_close_pressure"
    name = "Wavelet Open-Close Pressure"
    category = "microstructure"
    description = (
        "Compute a multi-horizon normalized open-to-close displacement using "
        "rolling volatility scales. The signal is strongest when the current "
        "bar closes near an extreme and recent return pressure is aligned across "
        "short and medium horizons."
    )
    window_length = 24
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        eps = 1e-12

        bar_range = (high - low).abs().replace(0.0, np.nan)
        oc_move = close - open_
        range_pos = ((close - low) / bar_range).clip(0.0, 1.0)
        close_extreme = (range_pos - 0.5) * 2.0

        ret1 = close.pct_change()
        oc_norm = oc_move / bar_range

        short_scale = ret1.rolling(4, min_periods=4).std()
        med_scale = ret1.rolling(12, min_periods=12).std()
        short_move = oc_norm / (short_scale.replace(0.0, np.nan) + eps)
        med_move = oc_norm / (med_scale.replace(0.0, np.nan) + eps)

        short_spectrum = short_move.rolling(3, min_periods=3).mean()
        med_spectrum = med_move.rolling(8, min_periods=8).mean()

        short_trend = ret1.rolling(3, min_periods=3).mean()
        med_trend = ret1.rolling(9, min_periods=9).mean()
        alignment = np.sign(short_trend.fillna(0.0) * med_trend.fillna(0.0))

        pressure = 0.45 * short_spectrum + 0.35 * med_spectrum + 0.20 * oc_norm
        spectral_strength = pressure * alignment

        extremity_boost = 1.0 + 0.75 * close_extreme.abs()
        continuation_bias = close_extreme * spectral_strength * extremity_boost

        sustained = (
            ret1.rolling(5, min_periods=5).sum().fillna(0.0)
            + ret1.rolling(10, min_periods=10).sum().fillna(0.0)
        )
        sustained = sustained / 2.0

        out = continuation_bias + 0.25 * sustained * alignment
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
