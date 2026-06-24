"""Markets often oscillate between coherent short-horizon continuation and noisy intrabar reversal regimes. This factor looks for stocks whose intrabar path is becoming more "mode-locked" across multiple horizons: when short-window close-open return consistency and longer-window range alignment both rise, the move is more likely to persist; when they diverge, the market is likely transitioning to a less stable state and the signal should fade. The inspiration is analogous to DMD/OPT-DMD in the paper: we want to isolate dominant, repeatable temporal patterns rather than the noisy residual path."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, delta, rank, ts_mean, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class MultiscaleCloseOpenAmbiguityShift(BaseFactor):
    """Multiscale close-open ambiguity shift emphasizing coherent path persistence."""

    factor_id = "multiscale_close_open_ambiguity_shift"
    name = "Multiscale Close-Open Ambiguity Shift"
    category = "other"
    description = (
        "Cross-sectional signal from agreement between short- and medium-horizon price path "
        "direction, combining rolling close-open returns with range-normalized intrabar path "
        "consistency."
    )
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        eps = 1e-12
        bar_range = (high - low).abs().replace(0.0, np.nan)
        body = (close - open_) / open_.replace(0.0, np.nan)
        path_position = ((close - low) / bar_range).clip(lower=0.0, upper=1.0)
        intrabar_direction = ((close - open_) / bar_range).clip(lower=-1.0, upper=1.0)

        short_consistency = ts_mean(np.sign(body).replace(0.0, np.nan).fillna(0.0), 5)
        medium_consistency = ts_mean(np.sign((close - open_) / (open_.abs() + eps)).replace(0.0, np.nan).fillna(0.0), 20)

        range_alignment = ts_mean((path_position - 0.5) * 2.0, 20)
        direction_strength = ts_mean(intrabar_direction.fillna(0.0), 10)

        short_rank = rank(short_consistency.fillna(0.0))
        medium_rank = rank(medium_consistency.fillna(0.0))
        alignment_rank = rank(range_alignment.fillna(0.0))
        direction_rank = rank(direction_strength.fillna(0.0))

        agreement = 0.5 * (short_rank + medium_rank)
        coherence = 0.5 * (alignment_rank + direction_rank)

        ambiguity = abs_(short_rank - alignment_rank)
        stability = 1.0 - ambiguity

        multiscale = 0.6 * agreement + 0.4 * coherence
        signal = multiscale * stability

        return signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
