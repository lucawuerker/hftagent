"""When a bar closes very near its high or low, short-horizon traders are often caught leaning into the move late, while liquidity providers may have already satisfied the imbalance inside the bar. A close that is strong relative to the full intrabar range but weak relative to the bar's start-to-end path can indicate an exhausted push that is more likely to revert on the next bar, especially in noisy markets where end-of-bar marking matters more than true continuation. This complements momentum-style seeds by focusing on failed intrabar persistence rather than raw directional strength."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, rank, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntrabarCloseToExtremeFade(BaseFactor):
    """Fade closes that finish near the intrabar extreme with volume-aware damping."""

    factor_id = "intrabar_close_to_extreme_fade"
    name = "Intrabar Close-to-Extreme Fade"
    category = "mean_reversion"
    description = (
        "Compute a contrarian score when close finishes near the bar extreme: "
        "compare close relative to the intrabar midpoint and normalize by range, "
        "then damp the signal on unusually high-volume bars."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        rng = (high - low).replace(0, np.nan)
        midpoint = 0.5 * (high + low)
        signed_mid_dist = (close - midpoint) / rng

        # Negative when closing near the high, positive when closing near the low.
        fade_core = -signed_mid_dist.clip(-1.0, 1.0)

        # Incorporate whether the close is weak relative to the bar's open-to-close path.
        oc_path = (close - open_).replace(0, np.nan)
        path_component = -((close - open_) / (high - low).replace(0, np.nan)).clip(-1.0, 1.0)
        path_component = path_component.where(oc_path.notna(), 0.0)

        raw_signal = 0.7 * fade_core + 0.3 * path_component

        # Dampen when volume is elevated versus recent average activity.
        vol_20 = adv(volume, 20)
        vol_ratio = (volume / vol_20.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        vol_damp = 1.0 / (1.0 + ts_mean(vol_ratio.fillna(1.0), 5).fillna(1.0).clip(lower=0.0))

        # Cross-sectional emphasis on the strongest extremes within each bar.
        extreme_rank = rank(fade_core.abs().fillna(0.0))
        signal = raw_signal * (0.5 + 0.5 * extreme_rank) * vol_damp

        return signal.fillna(0.0)
