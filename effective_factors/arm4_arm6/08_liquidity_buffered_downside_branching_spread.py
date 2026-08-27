"""Clusters of negative high-volume bars proxy self-exciting sell pressure: an initial loss raises the probability of additional forced selling through volatility targeting, margin constraints, and liquidity withdrawal. The same observed downside cluster should have very different consequences for firms with weak cash, poor quick-ratio coverage, and high debt versus firms that can absorb a temporary funding shock. The factor therefore shorts fragile names with an elevated downside branching estimate and buys buffered names experiencing comparable transient downside intensity. The mechanism is falsified if balance-sheet buffers do not separate continuation from recovery conditional on identical recent return-volume shock sequences."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


def _downside_branching_intensity(
    close: pd.DataFrame,
    volume: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate a causal, exponentially decayed downside-event intensity ratio."""
    safe_close = close.replace(0.0, np.nan)
    ret = (safe_close / safe_close.shift(1) - 1.0).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    safe_volume = volume.clip(lower=0.0).fillna(0.0)

    # Both thresholds use only information available before the current bar,
    # preventing an extreme current observation from relaxing its own trigger.
    prior_volatility = ret.shift(1).rolling(20, min_periods=15).std()
    prior_volume = safe_volume.shift(1).rolling(20, min_periods=15).mean()

    downside_excess = ((-ret / prior_volatility.replace(0.0, np.nan)) - 1.0)
    downside_excess = downside_excess.clip(lower=0.0, upper=8.0).fillna(0.0)
    volume_excess = (safe_volume / prior_volume.replace(0.0, np.nan) - 1.0)
    volume_excess = volume_excess.clip(lower=0.0, upper=8.0).fillna(0.0)

    # A continuous event mark preserves the severity of jointly unusual loss
    # and turnover instead of reducing the order-flow shock to a binary flag.
    event_mark = downside_excess * np.log1p(volume_excess)

    # This is the discrete exponential Hawkes kernel: recent marked sell events
    # retain influence but decay with a three-bar half-life.
    recent_intensity = event_mark.ewm(
        halflife=3.0, adjust=False, min_periods=1
    ).mean()

    # The long, lagged event rate is the name-specific exogenous/background
    # intensity.  Lagging it keeps the current offspring event out of baseline.
    baseline_intensity = event_mark.shift(1).rolling(60, min_periods=20).mean()
    branching_ratio = recent_intensity / baseline_intensity.replace(0.0, np.nan)

    return branching_ratio.replace([np.inf, -np.inf], np.nan)


def _fragility_composite(
    cash_ratio: pd.DataFrame,
    quick_ratio: pd.DataFrame,
    debt_to_equity: pd.DataFrame,
    interest_coverage: pd.DataFrame,
    free_cash_flow_yield: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-sectional funding fragility, where higher values mean weaker buffers."""
    def _neutral_rank(frame: pd.DataFrame) -> pd.DataFrame:
        return rank(frame.replace([np.inf, -np.inf], np.nan)).fillna(0.5)

    low_cash = 1.0 - _neutral_rank(cash_ratio)
    low_quick_coverage = 1.0 - _neutral_rank(quick_ratio)
    high_leverage = _neutral_rank(debt_to_equity)
    low_interest_coverage = 1.0 - _neutral_rank(interest_coverage)
    low_internal_funding = 1.0 - _neutral_rank(free_cash_flow_yield)

    return (
        low_cash
        + low_quick_coverage
        + high_leverage
        + low_interest_coverage
        + low_internal_funding
    ) / 5.0


@register_factor
class LiquidityBufferedDownsideBranchingSpread(BaseFactor):
    """Short fragile firms and buy buffered firms during downside sell-pressure cascades."""

    factor_id = "liquidity_buffered_downside_branching_spread"
    name = "Liquidity-Buffered Downside Branching Spread"
    category = "volatility"
    description = (
        "Causal exponential downside-event branching intensity, scaled by a "
        "cross-sectional balance-sheet fragility composite: elevated sell "
        "pressure is negative for weakly buffered firms and positive for "
        "well-buffered firms expected to recover from transient liquidation."
    )
    window_length = 80
    inputs = [
        "close",
        "volume",
        "cashRatio",
        "quickRatio",
        "debtToEquity",
        "interestCoverage",
        "freeCashFlowYield",
    ]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        branching_ratio = _downside_branching_intensity(close, data["volume"])
        fragility = _fragility_composite(
            data["cashRatio"],
            data["quickRatio"],
            data["debtToEquity"],
            data["interestCoverage"],
            data["freeCashFlowYield"],
        )

        # Ratios above one indicate excess recent offspring intensity relative
        # to background sell-event activity.  Zero intensity means no trade.
        activation = ((branching_ratio - 1.0) / 5.0).clip(
            lower=0.0, upper=1.0
        ).fillna(0.0)

        # Fragility above its neutral cross-sectional level receives a short
        # signal; firms with stronger liquidity buffers receive the opposite.
        signal = -activation * (fragility - 0.5)
        return signal.replace([np.inf, -np.inf], np.nan).fillna(0.0).reindex(
            index=close.index, columns=close.columns, fill_value=0.0
        )
