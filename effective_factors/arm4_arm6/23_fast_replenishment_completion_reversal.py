"""Fast-completion inventory reversal after liquidity replenishment.

Mechanism: alternating pressure is consistent with inventory transfer rather than
one-sided information.  A transient range-per-volume shock that normalizes within
three bars indicates that latent liquidity has replenished; fading the remaining
price displacement targets the short post-completion inventory unwind.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import indneutralize, rank
from quant_fund_agent.factors.registry import register_factor


def _fast_completion_state(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
) -> pd.DataFrame:
    """Causal proxy for a rapidly completed latent-liquidity recovery."""
    open_safe = open_.fillna(close)
    upper = pd.DataFrame(
        np.maximum(open_safe.to_numpy(dtype=float), close.to_numpy(dtype=float)),
        index=close.index,
        columns=close.columns,
    )
    lower = pd.DataFrame(
        np.minimum(open_safe.to_numpy(dtype=float), close.to_numpy(dtype=float)),
        index=close.index,
        columns=close.columns,
    )
    high_safe = high.fillna(upper).where(high.fillna(upper) >= upper, upper)
    low_safe = low.fillna(lower).where(low.fillna(lower) <= lower, lower)
    bar_range = (high_safe - low_safe).abs()

    endpoint = (close - open_safe) / bar_range.replace(0.0, np.nan)
    endpoint = endpoint.clip(lower=-1.0, upper=1.0).fillna(0.0)

    clean_volume = volume.clip(lower=0.0).fillna(0.0)
    volume_base = clean_volume.rolling(30, min_periods=15).median()
    relative_volume = clean_volume / volume_base.replace(0.0, np.nan)
    relative_volume = relative_volume.clip(lower=0.0, upper=12.0).fillna(0.0)
    pressure = endpoint * np.log1p(relative_volume)

    positive = (pressure > 0.0).astype(float).rolling(8, min_periods=6).mean()
    negative = (pressure < 0.0).astype(float).rolling(8, min_periods=6).mean()
    zero = (1.0 - positive - negative).clip(lower=0.0)
    entropy = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for probability in (positive, negative, zero):
        entropy = entropy - probability.where(probability > 0.0, 0.0) * np.log(
            probability.where(probability > 0.0, 1.0)
        )
    entropy = (entropy / np.log(3.0)).clip(lower=0.0, upper=1.0).fillna(0.0)

    gross_pressure = pressure.abs().rolling(8, min_periods=6).sum()
    net_pressure = pressure.rolling(8, min_periods=6).sum()
    cancellation = 1.0 - net_pressure.abs() / gross_pressure.replace(0.0, np.nan)
    competitive_flow = entropy * cancellation.clip(lower=0.0, upper=1.0).fillna(0.0)

    ret = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    impact = ret.abs() / np.sqrt(relative_volume.replace(0.0, np.nan))
    impact_base = impact.rolling(30, min_periods=15).median()
    shock = impact / impact_base.replace(0.0, np.nan) - 1.0

    recent_stress = shock.rolling(4, min_periods=3).max()
    recent_stress = (recent_stress.clip(lower=0.0, upper=3.0) / 3.0).fillna(0.0)

    impact_lag = impact.shift(3)
    decay_speed = (impact_lag - impact) / impact_lag.replace(0.0, np.nan)
    decay_speed = decay_speed.clip(lower=0.0, upper=1.0).fillna(0.0)

    current_excess = shock.clip(lower=0.0, upper=1.0).fillna(1.0)
    completion = (1.0 - current_excess).clip(lower=0.0, upper=1.0)
    return (competitive_flow * recent_stress * decay_speed * completion).fillna(0.0)


@register_factor
class FastReplenishmentCompletionReversal(BaseFactor):
    factor_id = "fast_replenishment_completion_reversal"
    name = "Fast Liquidity Replenishment Completion"
    category = "microstructure"
    description = (
        "Fade sector-relative three-day displacements only after alternating "
        "flow and a rapidly completed range-per-volume impact shock indicate "
        "liquidity replenishment."
    )
    window_length = 45
    inputs = ["open", "high", "low", "close", "volume", "sector"]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].replace(0.0, np.nan)
        state = _fast_completion_state(
            close,
            data["open"],
            data["high"],
            data["low"],
            data["volume"],
        )
        displacement = close.pct_change(3, fill_method=None)
        displacement = displacement.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        sector = data["sector"].fillna("Unknown")
        residual_displacement = indneutralize(displacement, sector).fillna(0.0)
        reversal = -residual_displacement * state
        reversal = indneutralize(reversal, sector).fillna(0.0)
        return (rank(reversal).fillna(0.5) - 0.5).fillna(0.0)
