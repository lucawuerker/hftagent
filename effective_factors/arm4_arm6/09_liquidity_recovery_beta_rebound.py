"""A market-wide deterioration in the price-impact-per-dollar-volume proxy identifies a latent liquidity shock, during which high-liquidity-beta names tend to suffer forced, non-informational selling. When that common shock begins to normalize, the prior relative displacement of the most exposed names should partially reverse as dealers and liquidity-sensitive investors rebuild inventory. The other side is liquidity-demanding investors that must reduce risk during stressed trading conditions, while arbitrage capital is slow because the recovery regime is uncertain. The hypothesis is falsified if high common-liquidity-beta losers continue to underperform after the aggregate impact shock has demonstrably eased."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


def _rolling_liquidity_beta(
    stock_returns: pd.DataFrame,
    liquidity_innovation: pd.Series,
    window: int,
    min_periods: int,
) -> pd.DataFrame:
    """Causal trailing beta of forced-selling returns to common liquidity shocks."""
    x_mean = liquidity_innovation.rolling(window, min_periods=min_periods).mean()
    x_centered = liquidity_innovation.sub(x_mean)
    x_var = x_centered.pow(2).rolling(window, min_periods=min_periods).mean()

    y = -stock_returns
    y_mean = y.rolling(window, min_periods=min_periods).mean()
    y_centered = y.sub(y_mean)
    covariance = y_centered.mul(x_centered, axis=0).rolling(
        window, min_periods=min_periods
    ).mean()

    beta = covariance.div(x_var.replace(0.0, np.nan), axis=0)
    return beta.replace([np.inf, -np.inf], np.nan)


@register_factor
class LiquidityRecoveryBetaRebound(BaseFactor):
    """Rebound signal for liquidity-beta losers after a common impact shock eases."""

    factor_id = "liquidity_recovery_beta_rebound"
    name = "Liquidity-Shock Beta Recovery"
    category = "statistical_arbitrage"
    description = (
        "Estimates each stock's trailing forced-selling beta to innovations in "
        "a cross-sectional Amihud-style impact proxy, then buys high-beta names "
        "with the largest shock-period losses only as the elevated aggregate "
        "liquidity shock begins to ease."
    )
    window_length = 80
    inputs = ["open", "close", "volume"]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        open_ = data["open"]
        volume = data["volume"]

        safe_open = open_.replace(0.0, np.nan)
        dollar_volume = (close.abs() * volume.abs()).replace(0.0, np.nan)
        intrabar_return = close.div(safe_open).sub(1.0)
        impact = intrabar_return.abs().div(dollar_volume)
        impact = impact.replace([np.inf, -np.inf], np.nan)

        # The cross-sectional median suppresses idiosyncratic single-name impact spikes.
        aggregate_impact = impact.median(axis=1, skipna=True).fillna(0.0)
        impact_baseline = aggregate_impact.rolling(60, min_periods=20).mean()
        relative_impact = aggregate_impact.div(
            impact_baseline.replace(0.0, np.nan)
        ).sub(1.0)
        relative_impact = relative_impact.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        liquidity_innovation = relative_impact.diff().fillna(0.0)
        innovation_scale = liquidity_innovation.rolling(20, min_periods=10).std()
        normalized_easing = (-liquidity_innovation).div(
            innovation_scale.replace(0.0, np.nan)
        )
        normalized_easing = normalized_easing.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Recovery requires both still-elevated impact and a statistically meaningful easing.
        shock_intensity = relative_impact.clip(lower=0.0, upper=3.0)
        easing_intensity = normalized_easing.clip(lower=0.0, upper=3.0)
        recovery_gate = (shock_intensity * easing_intensity).clip(upper=6.0)

        stock_returns = close.pct_change(fill_method=None)
        stock_returns = stock_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        beta = _rolling_liquidity_beta(
            stock_returns, liquidity_innovation, window=60, min_periods=40
        ).fillna(0.0)

        # Cumulative return while the common impact regime was stressed measures displacement.
        shock_weighted_returns = stock_returns.mul(shock_intensity, axis=0)
        displacement = shock_weighted_returns.rolling(12, min_periods=6).sum().fillna(0.0)

        beta_exposure = rank(beta).fillna(0.5)
        raw_signal = beta_exposure * (-displacement)
        cross_sectional_signal = rank(raw_signal).sub(0.5).fillna(0.0)

        active = recovery_gate.gt(0.0)
        return cross_sectional_signal.where(active, 0.0).reindex_like(close).fillna(0.0)
