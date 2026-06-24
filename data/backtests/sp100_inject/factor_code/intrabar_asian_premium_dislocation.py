"""The paper shows that short-maturity Asian-style payoffs are driven by the gap between the terminal price and the time-average, with the implied variance responding asymmetrically to how prices spend time above or below the average. In bar data, a ticker that finishes near the high but spends much of the bar below its own average often reflects delayed mean re-anchoring and short-lived price pressure rather than persistent demand. Conversely, a bar that closes weakly despite a strong average suggests distribution into strength and tends to underperform on a short horizon."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntrabarAsianPremiumDislocation(BaseFactor):
    """Cross-sectional dislocation factor based on intrabar close versus average price."""

    factor_id = "intrabar_asian_premium_dislocation"
    name = "Intrabar Asian Premium Dislocation"
    category = "volatility"
    description = (
        "Compute a normalized dislocation between the bar close and the bar "
        "average price, where the average is approximated by (open + high + "
        "low + close)/4, then scale it by the realized bar range and volume to "
        "emphasize persistent intrabar pricing rather than noisy closes. "
        "Cross-sectionally rank the negative of this dislocation so that names "
        "with closes above their intrabar average relative to range are shorts "
        "and names with closes below are longs."
    )
    window_length = 1
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"].fillna(0.0)
        high = data["high"].fillna(0.0)
        low = data["low"].fillna(0.0)
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        bar_avg = (open_ + high + low + close) * 0.25
        range_ = (high - low).abs().replace(0.0, np.nan)
        vol_scale = ts_mean(volume, 5).replace(0.0, np.nan)

        dislocation = (close - bar_avg) / range_
        dislocation = dislocation / vol_scale
        dislocation = dislocation.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return rank(-dislocation)
