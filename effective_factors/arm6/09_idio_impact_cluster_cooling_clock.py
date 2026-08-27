"""Idiosyncratic downside-impact clusters that have begun to cool.

Mechanism: a constrained seller generates clustered negative returns with elevated
price impact and participation.  Once the fast order-flow intensity falls below
its slower accumulated intensity, dealer inventory can be replenished and the
transient component of impact should reverse over the next several daily bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


def _sector_mean(values: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Causally broadcast the same-date sector cross-sectional mean."""
    result = pd.DataFrame(np.nan, index=values.index, columns=values.columns)
    for _, tickers in labels.groupby(labels).groups.items():
        block = values.loc[:, tickers]
        average = block.mean(axis=1)
        result.loc[:, tickers] = block.mul(0.0).add(average, axis=0)
    return result


def _trailing_zscore(values: pd.DataFrame, window: int) -> pd.DataFrame:
    """Trailing standardized surprise, with no information after date t."""
    mean = values.rolling(window, min_periods=window).mean()
    std = values.rolling(window, min_periods=window).std()
    return (values - mean) / std.replace(0.0, np.nan)


def _exp_kernel_intensity(events: pd.DataFrame, half_life: float) -> pd.DataFrame:
    """Causal exponentially decayed event intensity with comparable scales.

    This is the deterministic discrete-time Hawkes-kernel state without a
    fitted branching coefficient: each event adds to a state that decays at a
    fixed physical half-life.  Multiplication by (1-decay) makes intensities
    at different half-lives comparable under a stationary event rate.
    """
    decay = float(np.exp(-np.log(2.0) / half_life))
    event_values = events.fillna(0.0).to_numpy(dtype=float)
    state = np.zeros(event_values.shape[1], dtype=float)
    output = np.zeros_like(event_values, dtype=float)

    for row in range(event_values.shape[0]):
        state = decay * state + event_values[row]
        output[row] = (1.0 - decay) * state

    return pd.DataFrame(output, index=events.index, columns=events.columns)


@register_factor
class IdioImpactClusterCoolingClock(BaseFactor):
    factor_id = "idio_impact_cluster_cooling_clock"
    name = "Idiosyncratic Impact-Cluster Cooling Clock"
    category = "microstructure"
    description = (
        "Longs stocks where a recent stock-specific downside price-impact "
        "cluster is sufficiently large but its short exponential intensity is "
        "now below its slower accumulated intensity; sector-demeans the result."
    )
    window_length = 60
    inputs = ["close", "high", "low", "volume", "sector"]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].replace(0.0, np.nan)
        high = data["high"]
        low = data["low"]
        volume = data["volume"].abs().replace(0.0, np.nan)
        sector_labels = data["sector"].iloc[0].fillna("Unknown")

        ret = close.pct_change().clip(-0.25, 0.25)
        bar_range = (high - low).clip(lower=0.0)
        dollar_volume = (close.abs() * volume).replace(0.0, np.nan)
        impact = bar_range / dollar_volume
        log_impact = np.log(impact.where(impact > 0.0))

        # A forced-sale event requires all three ingredients: adverse price
        # movement, anomalous price impact, and unusually high participation.
        impact_surprise = _trailing_zscore(log_impact, 40).clip(-5.0, 5.0)
        log_volume = np.log(volume.where(volume > 0.0))
        volume_surprise = _trailing_zscore(log_volume, 40).clip(-4.0, 4.0)
        downside = (-ret).clip(lower=0.0)
        event = downside * impact_surprise.clip(lower=0.0) * (
            1.0 + volume_surprise.clip(lower=0.0)
        )

        # Remove broad sector de-risking: the target is a name-specific seller,
        # whose completion is more likely to leave a temporary impact debt.
        idio_event = (event - _sector_mean(event, sector_labels)).clip(lower=0.0)

        fast = _exp_kernel_intensity(idio_event, 2.0)
        slow = _exp_kernel_intensity(idio_event, 8.0)
        cooling = (slow - fast).clip(lower=0.0)

        # Completion is strongest when today's event is below the local fast
        # state, rather than merely when an old event happened long ago.
        cessation = ((fast - idio_event) / fast.replace(0.0, np.nan)).clip(0.0, 1.0)
        raw_signal = slow * cooling * cessation

        sector_neutral = raw_signal - _sector_mean(raw_signal, sector_labels)
        return rank(sector_neutral).fillna(0.5)
