"""The paper argues that short-horizon returns inherit a steep, asymmetric tail shape because volatility clusters after shocks and then decays nonlinearly, creating a persistent smile/skew effect. In a cross-sectional setting, names that repeatedly print large intrabar excursions relative to their close-to-open move are likely carrying elevated latent volatility-of-volatility, which tends to mean-revert in the next few bars. We therefore want to fade persistent amplitude-skew imbalances: stocks with unusually large high-low range versus net return should underperform on a short horizon, while compressed names should re-expand less aggressively."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class AmplitudeSkewPersistence(BaseFactor):
    """Short-horizon fade of persistent intrabar amplitude-skew imbalances."""

    factor_id = "amplitude_skew_persistence"
    name = "Amplitude Skew Persistence"
    category = "volatility"
    description = (
        "Compute a rolling difference between normalized intrabar amplitude "
        "and signed directional move, smooth it over a short window, and "
        "cross-sectionally rank it. High values indicate large shock-like "
        "excursions with weak directional commitment, which the factor treats "
        "as a short-term volatility overhang."
    )
    window_length = 8
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        safe_open = open_.replace(0, np.nan)
        amplitude = (high - low) / safe_open
        directional = (close - open_).abs() / safe_open
        skew_imbalance = amplitude - directional

        smoothed = ts_mean(skew_imbalance, 5)
        raw_signal = rank(smoothed)

        return raw_signal.reindex(index=close.index, columns=close.columns)
