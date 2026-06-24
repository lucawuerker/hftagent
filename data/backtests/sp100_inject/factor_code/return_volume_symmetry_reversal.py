"""Behavioral sentiment in the cited paper appears to react to returns rather than anticipate them, suggesting that short-horizon price moves can create temporary overreaction in participation. When price change and volume change are aligned in an unusually symmetric way, it often reflects crowd-driven confirmation rather than informed continuation; the move then tends to mean-revert as the sentiment impulse fades. This factor looks for that return-volume symmetry and fades it, aiming to capture post-impulse reversal rather than momentum."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, delta, decay_linear, ts_mean, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class ReturnVolumeSymmetryReversal(BaseFactor):
    """Fade short-horizon return-volume symmetry extremes."""

    factor_id = "return_volume_symmetry_reversal"
    name = "Return-Volume Symmetry Reversal"
    category = "sentiment"
    description = (
        "Compute a signed product of short-horizon return and volume change, "
        "then normalize by recent absolute return and volume activity. High "
        "positive values indicate a price move supported by unusually strong "
        "participation, while large negative values indicate capitulation-like "
        "moves; the signal is designed to fade the most sentiment-driven extremes."
    )
    window_length = 20
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]

        ret1 = close.pct_change()
        vol_chg = volume.pct_change()

        ret2 = delta(close, 2)
        vol2 = delta(volume, 2)

        symm = ret1 * vol_chg
        signed_symm = np.sign(symm) * np.sqrt(abs_(symm).fillna(0.0))

        abs_ret = ts_mean(abs_(ret1).fillna(0.0), 5)
        abs_vol = ts_mean(abs_(vol_chg).replace([np.inf, -np.inf], np.nan).fillna(0.0), 5)

        ret_activity = ts_sum(abs_(ret2).fillna(0.0), 5)
        vol_activity = ts_sum(abs_(vol2).fillna(0.0), 5)

        norm = decay_linear((abs_ret + abs_vol + ret_activity + vol_activity).fillna(0.0), 5)
        norm = norm.replace(0.0, np.nan)

        raw = -(signed_symm * (abs_ret + abs_vol)) / norm
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return raw
