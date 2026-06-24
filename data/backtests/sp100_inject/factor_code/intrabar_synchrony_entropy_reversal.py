"""When a stock’s bar path is dominated by a few large synchronized moves in both price and volume, that often reflects transient information shocks or forced trading rather than durable repricing. Drawing on the GDE intuition that continuous-time dynamics can distinguish smooth evolution from abrupt regime changes, this factor looks for bars where the intrabar price path is unusually coordinated with volume and then fades that coordination when it becomes too concentrated. The edge comes from the tendency of short-lived liquidity demand to mean-revert once the impulse is absorbed."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, delta, rank, returns, ts_mean, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntrabarSynchronyEntropyReversal(BaseFactor):
    """Fade sharp intrabar moves when price/volume synchrony is concentrated."""

    factor_id = "intrabar_synchrony_entropy_reversal"
    name = "Intrabar Synchrony Entropy Reversal"
    category = "other"
    description = (
        "Compute the absolute bar return times a synchrony score between signed "
        "intrabar movement and volume changes, then normalize by recent cross-"
        "sectional and time-series baselines. Long low-synchrony names after sharp "
        "moves; short high-synchrony names after sharp moves, with the signal "
        "attenuated when the bar path is already smooth."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        # Intrabar signed movement proxy: close-to-open return scaled by bar range direction.
        bar_ret = returns({"close": close})
        intrabar_move = (close - open_) / open_.replace(0, np.nan)
        range_span = (high - low).replace(0, np.nan)
        path_intensity = abs_(intrabar_move).where(range_span.notna(), 0.0)

        # Volume impulse and recent change in activity.
        vol_level = volume.replace(0, np.nan)
        vol_chg = delta(volume.fillna(0.0), 1)
        vol_chg_scale = ts_mean(abs_(vol_chg), 5).replace(0, np.nan)
        vol_surprise = abs_(vol_chg) / vol_chg_scale

        # Synchrony score: price move aligned with volume changes.
        signed_sync = (intrabar_move * vol_chg).where(intrabar_move.notna() & vol_chg.notna(), 0.0)
        sync_magnitude = abs_(signed_sync)
        sync_baseline = ts_mean(sync_magnitude, 10).replace(0, np.nan)
        sync_score = sync_magnitude / sync_baseline

        # Entropy-like concentration penalty: more concentrated activity => lower entropy.
        vol_share = volume / ts_sum(volume.fillna(0.0), 10).replace(0, np.nan)
        vol_share = vol_share.fillna(0.0).clip(lower=0.0)
        concentration = ts_sum(vol_share * vol_share, 10)
        entropy_proxy = 1.0 - concentration.fillna(0.0)

        # Cross-sectional baseline: sharp moves that are unusual across names.
        move_mag = abs_(bar_ret)
        cs_move = rank(move_mag)
        cs_sync = rank(sync_score.fillna(0.0))
        cs_entropy = rank(entropy_proxy.fillna(0.0))

        # Smooth-bar attenuation: if the path is smooth and low-range, reduce signal.
        smoothness = ts_mean((high - low) / close.replace(0, np.nan), 5).replace(0, np.nan)
        smoothness = smoothness.fillna(0.0)
        atten = 1.0 - rank(smoothness)

        raw = (
            (cs_move - 0.5)
            * (0.5 - cs_sync)
            * (0.5 + cs_entropy)
            * entropy_proxy.fillna(0.0)
            * atten.fillna(0.0)
        )

        # Final normalization by recent own-history baseline to stabilize scale.
        recent_abs = ts_mean(abs_(raw), 10).replace(0, np.nan)
        signal = raw / recent_abs

        signal = signal.where(np.isfinite(signal), 0.0)
        return signal.reindex(index=close.index, columns=close.columns).fillna(0.0)
