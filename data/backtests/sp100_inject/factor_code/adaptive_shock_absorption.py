"""BMM’s core insight is that markets alternate between quiet, convergent regimes and shock-like regimes where prices need to re-adapt quickly. In OHLCV data, a bar that closes back near its open after a large intrabar excursion and heavy turnover often indicates that the shock was absorbed by liquidity rather than followed through by informed trading. Such absorption should predict short-horizon continuation failure and mean reversion, especially when the candle’s range is large relative to the close displacement."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import adv, abs_, rank, sign


@register_factor
class AdaptiveShockAbsorption(BaseFactor):
    """Cross-sectional shock-absorption mean-reversion signal from OHLCV bars."""

    factor_id = "adaptive_shock_absorption"
    name = "Adaptive Shock Absorption"
    category = "microstructure"
    description = (
        "Compute a signed absorption score from intrabar range versus net "
        "close move, scaled by volume; large high-low range with small "
        "open-close displacement and elevated volume produces a strong "
        "positive mean-reversion signal."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)

        # Core bar geometry.
        intrabar_range = (high - low).abs()
        net_move = (close - open_).abs()
        absorption_ratio = intrabar_range / (net_move + 1e-12)

        # Heavy turnover relative to recent activity amplifies the signal.
        vol_rel = volume / (adv(volume, 20) + 1e-12)

        # Closing back near the open after a wide range is the classic absorbed-shock pattern.
        reversion_to_open = 1.0 - ((close - open_).abs() / (intrabar_range + 1e-12))
        reversion_to_open = reversion_to_open.clip(lower=-1.0, upper=1.0)

        # Prefer bars that explored a wide range but failed to trend away from the open.
        shock_intensity = rank(absorption_ratio)
        volume_intensity = rank(vol_rel)
        close_neutrality = rank(reversion_to_open)

        raw = shock_intensity * volume_intensity * close_neutrality

        # Positive mean-reversion signal for absorbed directional pressure.
        direction = -sign(close - open_)
        signal = raw * (1.0 + 0.5 * direction)

        # Keep output aligned and numeric; guard any residual NaNs.
        return signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
