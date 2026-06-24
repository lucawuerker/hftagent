"""When price closes consistently near the same side of its intraday body while the session range remains stable, it usually reflects orderly auction progression rather than noisy mean-reverting churn. In futures intraday data, persistent body placement combined with low instability in the bar structure can indicate informed directional commitment that is more reliable than raw return momentum, especially when simple sequential ML fails to extract signal from short bar sequences. This factor is designed to capture that non-sequential structural persistence using only OHLC geometry and volume-normalized range behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, adv, correlation, decay_linear, delta, rank, ts_mean, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class CloseBodyRangeStability(BaseFactor):
    """Body-placement persistence with range stability and moderate volume confirmation."""

    factor_id = "close_body_range_stability"
    name = "Close-Body Range Stability"
    category = "other"
    description = (
        "Compute a rolling stability score from the close's position within each bar's body/range "
        "and penalize large changes in that position over the last several bars. Multiply by a low-"
        "volatility adjustment based on recent intrabar range compression and moderate volume confirmation."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        body = (close - open_).abs()
        bar_range = (high - low).replace(0.0, np.nan)

        # Close location within the full bar range: near 1.0 implies closing near the high,
        # near -1.0 implies closing near the low.
        range_pos = ((close - low) - (high - close)) / bar_range
        range_pos = range_pos.clip(-1.0, 1.0).fillna(0.0)

        # Close location relative to the candle body, stabilizing when the close sits consistently
        # on one side of the body and avoiding division-by-zero on doji bars.
        body_pos = np.where(
            body.to_numpy(dtype=float) > 0.0,
            ((close - open_) / body.replace(0.0, np.nan)).clip(-1.0, 1.0),
            range_pos,
        )
        body_pos = pd.DataFrame(body_pos, index=close.index, columns=close.columns).fillna(0.0)

        # Stability: high when body position is persistent and changes only slowly.
        pos_change = abs_(delta(body_pos, 1)).fillna(0.0)
        persistence = ts_mean(1.0 - pos_change.clip(0.0, 1.0), 5).fillna(0.0)
        centered = ts_mean(body_pos, 5).fillna(0.0)
        centered_stability = (1.0 - abs_(body_pos - centered).fillna(0.0)).clip(0.0, 1.0)
        stability = (0.6 * persistence + 0.4 * centered_stability).clip(0.0, 1.0)

        # Penalize instability in the bar structure using recent range variability and range expansion.
        range_mean = ts_mean(bar_range, 10)
        range_std = bar_range.rolling(10, min_periods=10).std()
        range_cv = (range_std / range_mean.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        range_compression = (1.0 / (1.0 + range_cv)).clip(0.0, 1.0)

        range_change = abs_(delta(bar_range, 1)).fillna(0.0)
        range_change_norm = (range_change / range_mean.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        structure_penalty = (1.0 / (1.0 + range_change_norm)).clip(0.0, 1.0)

        # Moderate volume confirmation: prefer active bars, but avoid extreme spikes.
        vol_short = adv(volume, 5)
        vol_long = adv(volume, 20)
        vol_ratio = (vol_short / vol_long.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        vol_pref = 1.0 - abs_(vol_ratio - 1.0).clip(0.0, 2.0) / 2.0
        vol_pref = vol_pref.clip(0.0, 1.0)

        # Extra directional coherence: correlation between body position and volume over the short window.
        vol_body_corr = correlation(body_pos, volume, 5).fillna(0.0)
        vol_coherence = (0.5 + 0.5 * vol_body_corr.clip(-1.0, 1.0)).clip(0.0, 1.0)

        raw = stability * range_compression * structure_penalty * vol_pref * vol_coherence
        signal = decay_linear(raw.fillna(0.0), 3).fillna(0.0)

        return signal
