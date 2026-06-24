"""Markets often underreact when a bar’s path shifts from one intrabar regime to another, e.g., an early-range consolidation followed by a strong directional close. Inspired by the regime-switching literature of Hamilton and Kasahara-Shimotsu, this factor treats large open-to-midbar vs. midbar-to-close asymmetries as evidence of a latent state transition, and prices the post-transition continuation rather than the entire-bar return. It should work best when the bar’s breakout is accompanied by expanding volume, because the transition is more likely to reflect informed participation than a transient liquidity vacuum."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, adv, rank, returns, ts_mean, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class RegimeTransitionSurpriseMomentum(BaseFactor):
    """Signed intrabar regime-shift continuation signal scaled by range and volume."""

    factor_id = "regime_transition_surprise_momentum"
    name = "Regime Transition Surprise Momentum"
    category = "statistical_arbitrage"
    description = (
        "Measure the signed difference between the first-half and second-half returns "
        "within the bar, scaled by normalized range and volume. High values indicate a "
        "late-bar regime shift toward continuation; low values indicate an early-bar shock "
        "that is more likely to mean-revert."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        eps = 1e-12

        # First-half vs second-half intrabar return proxy.
        half_mid = 0.5 * (high + low)
        first_half_ret = (half_mid - open_) / open_.replace(0, np.nan)
        second_half_ret = (close - half_mid) / half_mid.replace(0, np.nan)
        regime_shift = second_half_ret - first_half_ret

        # Normalize by range and volume expansion.
        bar_range = (high - low).abs()
        normalized_range = bar_range / (open_.abs() + eps)
        volume_pressure = rank(volume / (adv(volume, 20) + eps))

        # Reinforce continuation when the close is moving in the direction of the shift.
        bar_return = returns(data)
        direction = np.sign(bar_return.fillna(0.0) + regime_shift.fillna(0.0))
        signed_shift = regime_shift * direction

        # Smooth the component scores and combine.
        shift_score = ts_mean(signed_shift.fillna(0.0), 5)
        range_score = rank(normalized_range.replace([np.inf, -np.inf], np.nan).fillna(0.0))
        volume_score = volume_pressure.fillna(0.0)

        raw = shift_score * (1.0 + range_score) * (0.5 + volume_score)
        penalty = 1.0 / (1.0 + ts_sum(abs_(bar_return).fillna(0.0), 10))
        signal = raw * penalty

        return signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
