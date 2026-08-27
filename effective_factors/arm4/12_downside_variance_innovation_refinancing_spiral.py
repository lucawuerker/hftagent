"""A negative return accompanied by an unusually large range relative to the stock's own recent downside-range distribution signals a downside volatility innovation rather than an ordinary drawdown. For firms with high debt burden, low cash coverage, and weak interest coverage, this can trigger mechanically procyclical de-risking by volatility-targeting investors, creditors, and risk managers, creating the crisis-style bad-news amplification emphasized by Community 7. The signal should fail if these balance-sheet characteristics do not differentiate the persistence of post-shock downside returns from that of equally large shocks in financially resilient firms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


def _downside_range_innovation(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    window: int,
    min_observations: int,
) -> pd.DataFrame:
    """Causal robust surprise in negative-return intrabar log ranges."""
    safe_high = high.where(high > 0.0)
    safe_low = low.where(low > 0.0)
    log_range = np.log(safe_high / safe_low).clip(0.0, None)

    log_return = np.log(close.where(close > 0.0) / close.shift(1).where(close.shift(1) > 0.0))
    current_downside_range = log_range.where(log_return < 0.0)

    # Shift before estimating the benchmark so today's range is not used to
    # define its own expected downside-risk distribution.
    prior_downside_range = current_downside_range.shift(1)
    median_range = prior_downside_range.rolling(
        window, min_periods=min_observations
    ).median()
    median_abs_deviation = prior_downside_range.sub(median_range).abs().rolling(
        window, min_periods=min_observations
    ).median()

    # A local range-based fallback preserves a finite scale where observed
    # downside ranges have temporarily clustered at the same rounded value.
    robust_scale = 1.4826 * median_abs_deviation
    fallback_scale = median_range.abs() * 0.10
    robust_scale = robust_scale.where(robust_scale > 1.0e-8, fallback_scale)
    robust_scale = robust_scale.where(robust_scale > 1.0e-8)

    innovation = (current_downside_range - median_range) / robust_scale
    innovation = innovation.where(
        current_downside_range.notna() & median_range.notna(), 0.0
    )
    return innovation.clip(0.0, 12.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _refinancing_fragility(
    net_debt_to_ebitda: pd.DataFrame,
    interest_coverage: pd.DataFrame,
    cash_ratio: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-sectional refinancing vulnerability from three PIT balance-sheet measures."""
    leverage = net_debt_to_ebitda.replace([np.inf, -np.inf], np.nan)
    coverage = interest_coverage.replace([np.inf, -np.inf], np.nan)
    liquidity = cash_ratio.replace([np.inf, -np.inf], np.nan)

    valid = leverage.notna() & coverage.notna() & liquidity.notna()

    # Negative EBITDA or EBIT coverage is treated as maximally weak rather
    # than allowing a negative reciprocal to masquerade as financial strength.
    leverage_pressure = leverage.clip(0.0, None)
    inverse_coverage = 1.0 / coverage.clip(0.05, None)
    inverse_cash = 1.0 / liquidity.clip(0.01, None)

    composite = (
        rank(leverage_pressure)
        + rank(inverse_coverage)
        + rank(inverse_cash)
    ) / 3.0
    fragility = rank(composite).where(valid)
    return fragility.replace([np.inf, -np.inf], np.nan)


@register_factor
class DownsideVarianceInnovationRefinancingSpiral(BaseFactor):
    """Short refinancing-fragile firms following exceptional downside-range shocks."""

    factor_id = "downside_variance_innovation_refinancing_spiral"
    name = "Refinancing-Exposed Downside Variance Spiral"
    category = "volatility"
    description = (
        "Robust downside realized-range innovations interacted with a cross-sectional "
        "refinancing-fragility composite of net debt to EBITDA, weak interest coverage, "
        "and weak cash coverage; high-fragility shock names receive negative scores."
    )
    window_length = 61
    inputs = [
        "close",
        "high",
        "low",
        "netDebtToEbitda",
        "interestCoverage",
        "cashRatio",
    ]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        innovation = _downside_range_innovation(
            close, data["high"], data["low"], 60, 12
        )
        fragility = _refinancing_fragility(
            data["netDebtToEbitda"],
            data["interestCoverage"],
            data["cashRatio"],
        )

        # Restrict the interaction to genuine positive range surprises.  The
        # signed exposure is positive for resilient firms and negative for
        # fragile firms experiencing comparable downside volatility shocks.
        shock_strength = rank(innovation.where(innovation > 0.0)).fillna(0.0)
        signed_resilience = 1.0 - 2.0 * fragility
        signal = shock_strength * signed_resilience

        return signal.reindex(index=close.index, columns=close.columns).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
