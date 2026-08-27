"""Low transition entropy in joint daily return-sign and relative-volume states is a proxy for structured, non-random execution, but is direction-neutral on its own. A fresh signed earnings or revenue-surprise innovation supplies the missing direction: when a low-entropy execution state follows a positive (negative) surprise and the initial price response is muted, institutional repricing should continue upward (downward) over the next week. The trade should fail if low entropy does not materially increase the conditional post-event move magnitude, or if surprise sign does not distinguish subsequent return direction; the likely counterparty is liquidity provision and discretionary investors who process the report more slowly than systematic event traders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


def _joint_execution_state(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Encode return-sign and trailing relative-volume quintile into 15 states."""
    safe_close = close.replace(0.0, np.nan)
    ret = safe_close.pct_change(fill_method=None)

    volume_valid = volume.where(volume > 0.0)
    volume_base = volume_valid.rolling(20, min_periods=10).mean()
    relative_volume = volume_valid / volume_base.replace(0.0, np.nan)

    q20 = relative_volume.rolling(30, min_periods=15).quantile(0.20)
    q40 = relative_volume.rolling(30, min_periods=15).quantile(0.40)
    q60 = relative_volume.rolling(30, min_periods=15).quantile(0.60)
    q80 = relative_volume.rolling(30, min_periods=15).quantile(0.80)

    volume_state = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    volume_state = volume_state.where(~(relative_volume <= q20), 0.0)
    volume_state = volume_state.where(~((relative_volume > q20) & (relative_volume <= q40)), 1.0)
    volume_state = volume_state.where(~((relative_volume > q40) & (relative_volume <= q60)), 2.0)
    volume_state = volume_state.where(~((relative_volume > q60) & (relative_volume <= q80)), 3.0)
    volume_state = volume_state.where(~(relative_volume > q80), 4.0)

    sign_state = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    sign_state = sign_state.where(~(ret < 0.0), 0.0)
    sign_state = sign_state.where(~(ret == 0.0), 1.0)
    sign_state = sign_state.where(~(ret > 0.0), 2.0)

    return sign_state * 5.0 + volume_state


def _markov_transition_entropy(states: pd.DataFrame, window: int) -> pd.DataFrame:
    """Trailing normalized conditional entropy of a 15-state Markov chain."""
    valid_transition = states.notna() & states.shift(1).notna()
    valid_count = valid_transition.astype(float).rolling(window, min_periods=window).sum()

    entropy = pd.DataFrame(0.0, index=states.index, columns=states.columns)
    total = valid_count.replace(0.0, np.nan)

    for previous_state in range(15):
        previous = states.shift(1).eq(float(previous_state)) & valid_transition
        outgoing = previous.astype(float).rolling(window, min_periods=window).sum()
        outgoing_safe = outgoing.replace(0.0, np.nan)

        for current_state in range(15):
            transition = previous & states.eq(float(current_state))
            count = transition.astype(float).rolling(window, min_periods=window).sum()
            probability_weight = count / total
            conditional_probability = count / outgoing_safe
            term = probability_weight * np.log(conditional_probability.where(count > 0.0))
            entropy = entropy - term.fillna(0.0)

    normalized = entropy / np.log(15.0)
    return normalized.where(valid_count >= 0.8 * window).clip(0.0, 1.0)


def _fresh_surprise_direction(
    eps_surprise: pd.DataFrame,
    revenue_surprise: pd.DataFrame,
) -> pd.DataFrame:
    """Create a signed impulse only when point-in-time surprise fields update."""
    eps_previous = eps_surprise.shift(1)
    revenue_previous = revenue_surprise.shift(1)

    eps_new = eps_surprise.notna() & (eps_previous.isna() | eps_surprise.ne(eps_previous))
    revenue_new = revenue_surprise.notna() & (
        revenue_previous.isna() | revenue_surprise.ne(revenue_previous)
    )

    eps_innovation = eps_surprise - eps_previous.where(eps_previous.notna(), 0.0)
    revenue_innovation = revenue_surprise - revenue_previous.where(revenue_previous.notna(), 0.0)

    eps_direction = np.sign(eps_innovation).where(eps_new, 0.0).fillna(0.0)
    revenue_direction = np.sign(revenue_innovation).where(revenue_new, 0.0).fillna(0.0)
    return 0.5 * eps_direction + 0.5 * revenue_direction


def _decayed_event_reserve(impulse: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Carry an event impulse forward causally with a short exponential decay."""
    reserve = pd.DataFrame(0.0, index=impulse.index, columns=impulse.columns)
    for lag in range(horizon):
        reserve = reserve + impulse.shift(lag).fillna(0.0) * np.exp(-lag / 3.0)
    return reserve


@register_factor
class EntropyDisambiguatedSurpriseImpactReserve(BaseFactor):
    """Signed post-surprise reserve conditioned on structured execution entropy."""

    factor_id = "entropy_disambiguated_surprise_impact_reserve"
    name = "Entropy-Disambiguated Surprise Impact Reserve"
    category = "microstructure"
    description = (
        "Cross-sectional rank of a fresh signed earnings or revenue surprise "
        "whose muted initial response is carried forward only while joint "
        "return-sign and relative-volume transition entropy remains low."
    )
    window_length = 60
    inputs = ["close", "volume", "epsSurprise", "revenueSurprise"]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        eps_surprise = data["epsSurprise"]
        revenue_surprise = data["revenueSurprise"]

        states = _joint_execution_state(close, volume)
        transition_entropy = _markov_transition_entropy(states, 35)
        structured_execution = (1.0 - transition_entropy).clip(0.0, 1.0)

        returns = close.replace(0.0, np.nan).pct_change(fill_method=None)
        return_volatility = returns.rolling(20, min_periods=10).std()
        impact_ratio = returns.abs() / return_volatility.replace(0.0, np.nan)
        muted_response = (1.0 - impact_ratio / 2.0).clip(0.0, 1.0).fillna(0.0)

        surprise_direction = _fresh_surprise_direction(eps_surprise, revenue_surprise)
        event_impulse = surprise_direction * muted_response
        carried_reserve = _decayed_event_reserve(event_impulse, 6)

        score = carried_reserve * structured_execution.fillna(0.0)
        return rank(score).fillna(0.5)
