"""The paper on price-limit changes shows that when markets become less stable, the added volatility is concentrated in low-frequency components rather than in noisy high-frequency flicker. In a live cross-section, stocks whose intrabar path is dominated by broad, smooth swings relative to their short-horizon noise are likely experiencing latent inventory risk and directional pressure that liquidity providers cannot easily fade. That makes them candidates for continuation on the same bar or the next bar, especially when the move is not merely a microstructure wiggle but a persistent intraday drift."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import abs_, delta, rank, signed_power, ts_mean


@register_factor
class IntrabarLowFrequencyVolatilityShift(BaseFactor):
    """Low-frequency intrabar volatility dominance signal."""

    factor_id = "intrabar_low_frequency_volatility_shift"
    name = "Intrabar Low-Frequency Volatility Shift"
    category = "volatility"
    description = (
        "Compute a low-frequency-to-high-frequency volatility proxy from "
        "intrabar returns by comparing smoothed intraday drift against "
        "short-horizon oscillation noise, then rank stocks cross-sectionally."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"].replace(0, np.nan)
        high = data["high"].replace(0, np.nan)
        low = data["low"].replace(0, np.nan)
        close = data["close"].replace(0, np.nan)

        # Intrabar directional movement: log(close/open), smoothed to capture
        # low-frequency drift across adjacent bars.
        intrabar_drift = np.log(close / open_)
        lf_drift = ts_mean(intrabar_drift, 5)

        # High-frequency intrabar noise: bar range relative to open, with a
        # short-term change component to emphasize flicker rather than trend.
        range_proxy = np.log(high / low)
        hf_noise = ts_mean(abs_(delta(range_proxy, 1)), 5)

        # Expose persistence by rewarding smooth drift and penalizing noisy
        # oscillation. signed_power keeps the signal robust while preserving sign.
        raw = lf_drift - hf_noise
        signal = signed_power(raw, 1.0)

        # Cross-sectional ranking into [0, 1] for comparability.
        out = rank(signal)
        return out.fillna(0.0)
