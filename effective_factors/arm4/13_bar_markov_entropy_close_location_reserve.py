"""Low entropy in a joint return-sign and relative-volume state sequence is a daily-bar proxy for mechanically structured execution rather than independent noise, following the hidden-order mechanism in the source paper. Entropy alone should forecast move magnitude, so direction is supplied only when the latest close is persistently located near the directional edge of its range, indicating that the structured flow has been accepted rather than rejected. The other side is liquidity providers and discretionary contrarians who accommodate a predictable execution program but cannot confidently infer its information sign until repeated close-location evidence accumulates. The mechanism is rejected if low-entropy episodes with persistent close-location do not exhibit stronger same-direction six-day returns than equally directional high-entropy episodes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


def _markov_transition_entropy(
    state: pd.DataFrame,
    n: int,
    n_states: int,
    min_transitions: int,
) -> pd.DataFrame:
    """Causal rolling conditional entropy of a finite-state Markov process.

    The estimate is the empirical entropy rate
    -sum_i,j p(i,j) log(p(j | i)) over transitions observed in each trailing
    window. Invalid state observations do not contribute transitions.
    """
    previous = state.shift(1)
    valid = state.notna() & previous.notna()
    total = valid.astype(float).rolling(n, min_periods=1).sum()
    total_safe = total.replace(0.0, np.nan)
    entropy = pd.DataFrame(0.0, index=state.index, columns=state.columns)

    for origin in range(n_states):
        origin_mask = previous.eq(float(origin)) & valid
        origin_count = origin_mask.astype(float).rolling(n, min_periods=1).sum()
        origin_safe = origin_count.replace(0.0, np.nan)

        for destination in range(n_states):
            transition_mask = origin_mask & state.eq(float(destination))
            count = transition_mask.astype(float).rolling(n, min_periods=1).sum()
            probability = (count / origin_safe).where(count > 0.0)
            joint_weight = count / total_safe
            entropy = entropy - (joint_weight * np.log(probability)).fillna(0.0)

    return entropy.where(total >= float(min_transitions))


def _joint_return_volume_state(
    close: pd.DataFrame,
    volume: pd.DataFrame,
) -> pd.DataFrame:
    """Build 15 states from return sign and trailing relative-volume quintile."""
    ret = close.pct_change()
    volume_base = volume.rolling(20, min_periods=10).mean().replace(0.0, np.nan)
    relative_volume = volume / volume_base
    volume_percentile = relative_volume.rolling(60, min_periods=20).rank(pct=True)

    volume_quintile = np.floor((volume_percentile * 5.0).clip(upper=4.999999))
    return_sign = pd.DataFrame(
        np.where(ret.gt(0.0), 2.0, np.where(ret.lt(0.0), 0.0, 1.0)),
        index=close.index,
        columns=close.columns,
    )
    valid = ret.notna() & volume_quintile.notna()
    return (return_sign * 5.0 + volume_quintile).where(valid)


@register_factor
class StructuredFlowCloseLocationReserve(BaseFactor):
    """Low-entropy execution reserve signed by accepted close-location pressure."""

    factor_id = "bar_markov_entropy_close_location_reserve"
    name = "Structured Flow Close-Location Reserve"
    category = "microstructure"
    description = (
        "For each ticker, discretize daily return sign and rolling relative-volume "
        "quintile into a finite state process, estimate causal rolling transition "
        "entropy, and combine low entropy with signed close-location value and "
        "recent directional persistence. Rank the resulting signed reserve "
        "cross-sectionally each day."
    )
    window_length = 120
    inputs = ["close", "high", "low", "volume"]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        state = _joint_return_volume_state(close, volume)
        entropy = _markov_transition_entropy(
            state=state,
            n=60,
            n_states=15,
            min_transitions=30,
        )
        entropy_reserve = (1.0 - entropy / np.log(15.0)).clip(lower=0.0, upper=1.0)

        bar_range = (high - low).replace(0.0, np.nan)
        close_location = ((2.0 * close - high - low) / bar_range).clip(-1.0, 1.0)
        location_persistence = close_location.rolling(5, min_periods=3).mean()

        ret = close.pct_change()
        directional_persistence = np.sign(ret).rolling(5, min_periods=3).mean()
        agreement = (
            np.sign(location_persistence) * directional_persistence
        ).clip(-1.0, 1.0)
        accepted_direction = location_persistence * (0.5 + 0.5 * agreement)

        reserve = entropy_reserve * accepted_direction
        return rank(reserve.fillna(0.0)).fillna(0.5)
