"""Daily OHLCV can reveal a latent transition from a replenished market to a liquidity-withdrawal state even without visible order-book depth: range expands, volume becomes abnormal, and closes migrate persistently toward one bar edge. Once that transition occurs, the initial move can be underreacted because risk-constrained liquidity providers widen participation gradually and investors needing immediacy continue to trade. The signal is falsified if posterior withdrawal transitions forecast only symmetric volatility rather than a signed continuation conditional on close-location asymmetry, or if their returns disappear after matching on range, turnover, and recent return."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


def _causal_zscore(frame: pd.DataFrame, span: int) -> pd.DataFrame:
    """Standardize against a strictly prior exponentially weighted history."""
    prior_mean = frame.ewm(span=span, adjust=False, min_periods=5).mean().shift(1)
    prior_std = frame.ewm(span=span, adjust=False, min_periods=5).std().shift(1)
    zscore = (frame - prior_mean) / prior_std.replace(0.0, np.nan)
    return zscore.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-4.0, 4.0)


def _withdrawal_transition_probability(features: list[pd.DataFrame]) -> pd.DataFrame:
    """Causal fixed-parameter three-state HMM filter for latent liquidity state.

    States are replenished, absorbed, and withdrawing.  The transition matrix
    makes withdrawal comparatively persistent but permits a fresh withdrawal
    transition after an abnormal range/volume/edge-concentration observation.
    """
    index = features[0].index
    columns = features[0].columns
    observations = np.stack(
        [feature.fillna(0.0).to_numpy(dtype=float) for feature in features], axis=2
    )
    observations = np.nan_to_num(observations, nan=0.0, posinf=4.0, neginf=-4.0)

    n_time, n_assets, _ = observations.shape
    transition = np.array(
        [
            [0.955, 0.035, 0.010],
            [0.105, 0.840, 0.055],
            [0.035, 0.115, 0.850],
        ],
        dtype=float,
    )
    means = np.array(
        [
            [-0.35, -0.25, -0.20, -0.15],
            [0.60, 0.65, -0.55, -0.55],
            [0.95, 0.95, 0.85, 0.85],
        ],
        dtype=float,
    )
    variances = np.array(
        [
            [0.95, 0.95, 1.10, 1.10],
            [1.20, 1.20, 0.85, 0.85],
            [1.35, 1.35, 0.90, 0.90],
        ],
        dtype=float,
    )

    posterior = np.tile(np.array([0.75, 0.20, 0.05]), (n_assets, 1))
    fresh_withdrawal = np.zeros((n_time, n_assets), dtype=float)

    for t in range(n_time):
        previous = posterior
        predicted = previous @ transition
        innovation = observations[t, :, None, :] - means[None, :, :]
        log_likelihood = -0.5 * np.sum(
            (innovation * innovation) / variances[None, :, :]
            + np.log(variances[None, :, :]),
            axis=2,
        )
        log_likelihood -= log_likelihood.max(axis=1, keepdims=True)
        likelihood = np.exp(log_likelihood)

        unnormalized = predicted * likelihood
        normalizer = unnormalized.sum(axis=1, keepdims=True)
        normalizer = np.where(normalizer > 0.0, normalizer, 1.0)
        posterior = unnormalized / normalizer

        # Joint filtered probability that today's withdrawal state was entered
        # from either non-withdrawal state rather than merely persisted.
        fresh_prior = (
            previous[:, 0] * transition[0, 2]
            + previous[:, 1] * transition[1, 2]
        )
        fresh_withdrawal[t] = (fresh_prior * likelihood[:, 2]) / normalizer[:, 0]

    return pd.DataFrame(fresh_withdrawal, index=index, columns=columns)


@register_factor
class LatentLiquidityEscapeVector(BaseFactor):
    """Fresh hidden liquidity-withdrawal transitions aligned with directional impact."""

    factor_id = "latent_liquidity_escape_vector"
    name = "Latent Liquidity Escape Vector"
    category = "statistical_arbitrage"
    description = (
        "Cross-sectional rank of a causal three-state hidden-liquidity filter's "
        "fresh withdrawal-transition probability, signed by close-location "
        "innovation and weighted by recent net price-impact efficiency."
    )
    window_length = 30
    inputs = ["open", "high", "low", "close", "volume"]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_price = data["open"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)

        bar_range = (high - low).abs()
        prior_close = close.shift(1).abs().replace(0.0, np.nan)
        normalized_range = (bar_range / prior_close).replace([np.inf, -np.inf], np.nan)

        safe_range = bar_range.replace(0.0, np.nan)
        body_to_range = ((close - open_price).abs() / safe_range).clip(0.0, 1.0)
        close_location = ((2.0 * close - high - low) / safe_range).clip(-1.0, 1.0)

        log_volume = np.log1p(volume.clip(lower=0.0))
        range_z = _causal_zscore(normalized_range.fillna(0.0), 20)
        volume_z = _causal_zscore(log_volume.fillna(0.0), 20)
        body_z = _causal_zscore(body_to_range.fillna(0.0), 20)
        edge_z = _causal_zscore(close_location.abs().fillna(0.0), 20)

        fresh_withdrawal = _withdrawal_transition_probability(
            [range_z, volume_z, body_z, edge_z]
        )

        prior_location = (
            close_location.fillna(0.0)
            .ewm(span=10, adjust=False, min_periods=3)
            .mean()
            .shift(1)
        )
        location_innovation = (
            close_location.fillna(0.0) - prior_location
        ).fillna(0.0).clip(-2.0, 2.0)

        returns = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        net_move = returns.rolling(5, min_periods=3).sum()
        gross_move = returns.abs().rolling(5, min_periods=3).sum()
        impact_efficiency = (
            net_move.abs() / gross_move.replace(0.0, np.nan)
        ).fillna(0.0).clip(0.0, 1.0)

        signal = fresh_withdrawal * location_innovation * impact_efficiency
        return rank(signal.fillna(0.0))
