"""Speculative bubbles often exhibit accelerating price advances interspersed with increasingly rapid drawdowns as leveraged or attention-driven demand approaches a critical transition. The signal shorts names with a statistically significant log-periodic acceleration pattern, an extended valuation multiple, and weak cash-generation support; it can be positive for similarly volatile names whose cash-flow quality invalidates the bubble interpretation. The expected counterparty is momentum, retail-attention, and benchmarked growth capital that extrapolates the visible price trajectory while underweighting financing and cash-realization constraints. The hypothesis is falsified if critical-path diagnostics do not predict relatively negative six-day returns after controlling for ordinary trend and volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


def _trailing_weighted_sum(
    values: pd.DataFrame, weights: np.ndarray, window: int
) -> pd.DataFrame:
    """Apply fixed trailing regression weights independently to each column."""
    return values.rolling(window, min_periods=window).apply(
        lambda sample: float(np.dot(sample, weights)), raw=True
    )


def _logperiodic_phase_exit(
    close: pd.DataFrame, window: int = 60, endpoint_days: float = 6.0
) -> pd.DataFrame:
    """Causal fixed-endpoint LPPL diagnostic estimated separately at each bar.

    The fitted basis is a quadratic log-price trend plus sin/cos terms in
    log(time-to-endpoint).  Because the basis is fixed relative to each
    trailing window, rolling normal-equation products give a causal OLS fit.
    """
    clean_close = close.where(close > 0.0).replace([np.inf, -np.inf], np.nan)
    log_price = np.log(clean_close)

    # Relative ages in the current trailing window, with zero denoting today.
    age = np.arange(-(window - 1), 1, dtype=float)
    time_to_endpoint = endpoint_days - age
    omega = 7.0
    phase_path = omega * np.log(time_to_endpoint)

    design = np.column_stack(
        [
            np.ones(window),
            age / window,
            (age / window) ** 2,
            np.sin(phase_path),
            np.cos(phase_path),
        ]
    )
    gram_inverse = np.linalg.pinv(design.T @ design)

    # X'y for every trailing window.  Each rolling apply is columnwise and
    # only consumes the contemporaneous and prior window observations.
    cross_products = [
        _trailing_weighted_sum(log_price, design[:, j], window)
        for j in range(design.shape[1])
    ]

    coefficients = []
    for row in range(design.shape[1]):
        coefficient = cross_products[0] * 0.0
        for col in range(design.shape[1]):
            coefficient = coefficient + cross_products[col] * gram_inverse[row, col]
        coefficients.append(coefficient)

    beta_quadratic = coefficients[2]
    beta_sin = coefficients[3]
    beta_cos = coefficients[4]

    # OLS residual variance follows SSE = y'y - beta'X'y.  This avoids a
    # non-causal refit of historical coefficients for every output row.
    y_squared = (log_price * log_price).rolling(window, min_periods=window).sum()
    fitted_cross_product = cross_products[0] * 0.0
    for coefficient, product_value in zip(coefficients, cross_products):
        fitted_cross_product = fitted_cross_product + coefficient * product_value
    sse = (y_squared - fitted_cross_product).clip(lower=0.0)
    residual_variance = sse / float(window - design.shape[1])

    amplitude = np.sqrt(beta_sin * beta_sin + beta_cos * beta_cos)
    sinusoid_variance_scale = 0.5 * (
        float(gram_inverse[3, 3]) + float(gram_inverse[4, 4])
    )
    amplitude_se = np.sqrt(residual_variance * sinusoid_variance_scale).replace(
        0.0, np.nan
    )
    amplitude_tstat = amplitude / amplitude_se

    # Derivative of a*sin(theta)+b*cos(theta), theta=omega*log(t_c-t).
    endpoint_phase = omega * np.log(endpoint_days)
    oscillation_slope = (omega / endpoint_days) * (
        -beta_sin * np.cos(endpoint_phase) + beta_cos * np.sin(endpoint_phase)
    )
    bearish_phase = (-endpoint_days * oscillation_slope / amplitude.replace(0.0, np.nan))
    bearish_phase = bearish_phase.clip(lower=0.0, upper=2.0) / 2.0

    # The normalized quadratic coefficient is the fitted acceleration across
    # the complete window, rather than an ordinary linear momentum measure.
    acceleration = (beta_quadratic * 4.0).clip(lower=0.0)
    acceleration_gate = np.tanh(acceleration / 0.08)
    significance_gate = np.tanh(amplitude_tstat.clip(lower=0.0) / 3.0)

    return (bearish_phase * acceleration_gate * significance_gate).replace(
        [np.inf, -np.inf], np.nan
    )


def _neutral_rank(values: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank with a neutral value for unavailable fundamentals."""
    return rank(values.replace([np.inf, -np.inf], np.nan)).fillna(0.5)


@register_factor
class CashVetoedLogperiodicPhaseExit(BaseFactor):
    """Bearish log-periodic bubble-exit score vetoed by cash-generation quality."""

    factor_id = "cash_vetoed_logperiodic_phase_exit"
    name = "Cash-Vetoed Log-Periodic Phase Exit"
    category = "mean_reversion"
    description = (
        "Fits a causal accelerating log-price and log-periodic phase model over "
        "60 bars, then ranks expected six-day returns after scaling phase-exit "
        "risk by valuation stretch and vetoing it with cash-flow support."
    )
    window_length = 60
    inputs = [
        "close",
        "evToSales",
        "psRatio",
        "freeCashFlowYield",
        "incomeQuality",
        "cash",
        "totalDebt",
    ]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        phase_exit = _logperiodic_phase_exit(close)

        ev_to_sales = data["evToSales"].where(data["evToSales"] > 0.0)
        price_to_sales = data["psRatio"].where(data["psRatio"] > 0.0)
        valuation_stretch = (
            _neutral_rank(ev_to_sales) + _neutral_rank(price_to_sales)
        ) / 2.0

        fcf_support = _neutral_rank(data["freeCashFlowYield"])
        income_support = _neutral_rank(data["incomeQuality"])

        cash = data["cash"].clip(lower=0.0)
        debt = data["totalDebt"].abs().clip(lower=0.0)
        cash_to_capital = cash / (cash + debt).replace(0.0, np.nan)
        balance_sheet_support = _neutral_rank(cash_to_capital)
        fundamental_support = (
            fcf_support + income_support + balance_sheet_support
        ) / 3.0

        # Strong cash realization both suppresses the short thesis and supplies
        # a modest positive offset for volatile but fundamentally supported names.
        bubble_penalty = phase_exit.fillna(0.0) * valuation_stretch * (
            1.0 - 0.85 * fundamental_support
        )
        expected_return_score = 0.25 * fundamental_support - bubble_penalty

        return _neutral_rank(expected_return_score).reindex(
            index=close.index, columns=close.columns, fill_value=0.5
        )
