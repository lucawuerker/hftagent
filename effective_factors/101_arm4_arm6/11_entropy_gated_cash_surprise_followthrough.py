"""The source paper argues that low order-flow entropy identifies structured, potentially informed activity but cannot determine direction by itself. Direction is supplied here by a fresh, cash-validated earnings surprise: positive reported EPS or revenue surprises that coincide with improving operating-cash-flow quality are more credible when the subsequent daily price-volume state sequence becomes unusually predictable. The edge should disappear if low-entropy paths do not selectively amplify post-report continuation relative to equally sized surprises in high-entropy paths; the other side is investors who either trade the headline mechanically or wait for subsequent confirmation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


def _cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Robust per-date standardization, with sparse cross sections made neutral."""
    clean = df.replace([np.inf, -np.inf], np.nan)
    mean = clean.mean(axis=1)
    scale = clean.std(axis=1).replace(0.0, np.nan)
    return clean.sub(mean, axis=0).div(scale, axis=0).clip(-4.0, 4.0).fillna(0.0)


def _rolling_sum_array(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling sums along time without any future observations."""
    n_time, n_assets = values.shape
    out = np.zeros((n_time, n_assets), dtype=float)
    padded = np.vstack((np.zeros((1, n_assets), dtype=float), np.cumsum(values, axis=0)))
    if n_time >= window:
        out[window - 1 :] = padded[window:] - padded[:-window]
    return out


def _markov_transition_entropy(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    transition_window: int,
) -> pd.DataFrame:
    """Causal normalized entropy rate of nine return-sign/volume states.

    A state is the Cartesian product of return sign (down, flat, up) and
    relative-volume bucket (low, normal, high).  The entropy is estimated from
    the trailing empirical transition matrix, not merely from state frequencies.
    """
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
    volume_mean = volume.rolling(20, min_periods=10).mean().replace(0.0, np.nan)
    relative_volume = volume.div(volume_mean).replace([np.inf, -np.inf], np.nan)

    valid = ret.notna() & relative_volume.notna() & volume.notna()
    ret_values = ret.to_numpy(dtype=float)
    rv_values = relative_volume.to_numpy(dtype=float)
    valid_values = valid.to_numpy(dtype=bool)

    sign_state = np.where(ret_values > 0.0, 2, np.where(ret_values < 0.0, 0, 1))
    volume_state = np.where(rv_values < 0.75, 0, np.where(rv_values > 1.25, 2, 1))
    states = sign_state * 3 + volume_state
    states[~valid_values] = -1

    n_time, n_assets = states.shape
    transition_valid = np.zeros((n_time, n_assets), dtype=bool)
    if n_time > 1:
        transition_valid[1:] = (states[1:] >= 0) & (states[:-1] >= 0)

    total_transitions = _rolling_sum_array(transition_valid.astype(float), transition_window)
    entropy = np.zeros((n_time, n_assets), dtype=float)

    for origin in range(9):
        origin_mask = np.zeros((n_time, n_assets), dtype=bool)
        if n_time > 1:
            origin_mask[1:] = transition_valid[1:] & (states[:-1] == origin)
        origin_count = _rolling_sum_array(origin_mask.astype(float), transition_window)
        safe_origin_count = np.where(origin_count > 0.0, origin_count, 1.0)

        for destination in range(9):
            pair_mask = np.zeros((n_time, n_assets), dtype=bool)
            if n_time > 1:
                pair_mask[1:] = origin_mask[1:] & (states[1:] == destination)
            pair_count = _rolling_sum_array(pair_mask.astype(float), transition_window)
            positive = pair_count > 0.0
            contribution = np.zeros_like(pair_count)
            contribution[positive] = -pair_count[positive] * np.log(
                pair_count[positive] / safe_origin_count[positive]
            )
            entropy += contribution

    safe_total = np.where(total_transitions > 0.0, total_transitions, 1.0)
    entropy = entropy / safe_total / np.log(9.0)
    entropy[total_transitions < float(transition_window)] = np.nan
    return pd.DataFrame(entropy, index=close.index, columns=close.columns)


def _expit_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Numerically stable logistic gate for DataFrame inputs."""
    clipped = df.clip(-12.0, 12.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@register_factor
class EntropyGatedCashSurpriseFollowthrough(BaseFactor):
    """Low-entropy price-volume paths amplify credible underreacted cash surprises."""

    factor_id = "entropy_gated_cash_surprise_followthrough"
    name = "Low-Entropy Cash-Surprise Follow-Through"
    category = "microstructure"
    description = (
        "Construct a causal daily Markov-transition entropy from discretized "
        "return-sign and relative-volume states over a rolling window. Multiply "
        "the low-entropy percentile deficit by a decayed, cross-sectionally "
        "standardized signed earnings/revenue surprise, with cash-flow quality "
        "and the initial price-response shortfall used as reliability and "
        "underreaction gates."
    )
    window_length = 64
    inputs = [
        "close",
        "volume",
        "epsSurprise",
        "revenueSurprise",
        "operatingCashFlow",
        "totalAssets",
        "incomeQuality",
    ]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"].fillna(0.0)
        eps_surprise = data["epsSurprise"]
        revenue_surprise = data["revenueSurprise"]

        entropy = _markov_transition_entropy(close, volume, 30)
        low_entropy_deficit = (1.0 - rank(entropy)).fillna(0.0)

        eps_changed = eps_surprise.notna() & (
            eps_surprise.shift(1).isna() | eps_surprise.ne(eps_surprise.shift(1))
        )
        revenue_changed = revenue_surprise.notna() & (
            revenue_surprise.shift(1).isna() | revenue_surprise.ne(revenue_surprise.shift(1))
        )
        fresh_report = eps_changed | revenue_changed

        eps_z = _cross_sectional_zscore(eps_surprise).where(eps_surprise.notna())
        revenue_z = _cross_sectional_zscore(revenue_surprise).where(revenue_surprise.notna())
        available_count = eps_z.notna().astype(float) + revenue_z.notna().astype(float)
        surprise = (eps_z.fillna(0.0) + revenue_z.fillna(0.0)).div(
            available_count.replace(0.0, np.nan)
        ).fillna(0.0)

        assets = data["totalAssets"].replace(0.0, np.nan)
        ocf_yield = data["operatingCashFlow"].div(assets).replace([np.inf, -np.inf], np.nan)
        reported_quality = data["incomeQuality"].replace([np.inf, -np.inf], np.nan)
        cash_quality = reported_quality.where(reported_quality.notna(), ocf_yield)
        quality_change = cash_quality.diff().replace([np.inf, -np.inf], np.nan)
        cash_gate = _expit_frame(_cross_sectional_zscore(quality_change))

        daily_return = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        response_volatility = daily_return.rolling(20, min_periods=10).std().replace(0.0, np.nan)
        standardized_response = daily_return.div(response_volatility).fillna(0.0)
        direction = np.sign(surprise)
        underreaction_gate = _expit_frame(-direction * standardized_response)

        event_impulse = (
            surprise
            * cash_gate
            * underreaction_gate
            * fresh_report.astype(float)
        )
        decayed_surprise = event_impulse.ewm(halflife=3.0, adjust=False, min_periods=1).mean()

        signal = (low_entropy_deficit * decayed_surprise).replace([np.inf, -np.inf], np.nan)
        return rank(signal).fillna(0.5)
