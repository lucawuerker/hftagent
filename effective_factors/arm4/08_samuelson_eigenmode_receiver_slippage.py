"""A sector-wide shock is transmitted unevenly because liquidity, mandate constraints, and stock-specific attention determine which constituents adjust immediately. Estimate the dominant rolling sector return eigenmode, then buy high-loading stocks whose contemporaneous response is below the mode-implied move and short those whose response is above it. The mechanism is falsified if residual response gaps close contemporaneously rather than over the following week, or if they have no relation to stable pre-shock sector loadings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


def _pre_shock_eigenmode_residual(
    ret: pd.DataFrame,
    sector: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Compute causal within-sector PCA residuals using only returns before t.

    For each date and sector, the covariance matrix is estimated from the
    preceding window.  The contemporaneous, demeaned sector return vector is
    projected onto its leading pre-shock covariance eigenvector.  The residual
    is the part of the current move not explained by that established mode.
    """
    n_rows, n_cols = ret.shape
    result = np.full((n_rows, n_cols), np.nan, dtype=float)
    returns_array = ret.to_numpy(dtype=float)
    sector_array = sector.to_numpy(dtype=object)

    min_history = max(20, int(0.8 * window))

    for t in range(window, n_rows):
        labels = sector_array[t]
        groups: dict[object, list[int]] = {}
        for col_idx, label in enumerate(labels):
            if pd.isna(label):
                continue
            try:
                groups.setdefault(label, []).append(col_idx)
            except TypeError:
                # Defensive fallback for an unexpected non-hashable label.
                groups.setdefault(str(label), []).append(col_idx)

        for members in groups.values():
            if len(members) < 3:
                continue

            member_idx = np.asarray(members, dtype=int)
            history = returns_array[t - window:t, member_idx]
            observed = np.isfinite(history)
            counts = observed.sum(axis=0)
            eligible = counts >= min_history

            if eligible.sum() < 3:
                continue

            active_idx = member_idx[eligible]
            history = history[:, eligible]
            observed = observed[:, eligible]

            sums = np.where(observed, history, 0.0).sum(axis=0)
            means = sums / counts[eligible]
            centered_history = np.where(observed, history - means, 0.0)
            covariance = centered_history.T @ centered_history / float(window - 1)
            covariance = (covariance + covariance.T) * 0.5

            try:
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            except np.linalg.LinAlgError:
                continue

            leading_value = float(eigenvalues[-1])
            total_variance = float(np.trace(covariance))
            if not np.isfinite(leading_value) or total_variance <= 0.0:
                continue

            leading_share = leading_value / total_variance
            n_active = len(active_idx)

            # A leading mode must exceed the equal-variance benchmark by a
            # meaningful amount before interpreting a common shock as present.
            if leading_share <= (1.0 / n_active + 0.10):
                continue

            mode = eigenvectors[:, -1]
            current = returns_array[t, active_idx]
            current_observed = np.isfinite(current)
            if current_observed.sum() < 2:
                continue

            current_centered = current - means
            mode_observed = mode[current_observed]
            denominator = float(mode_observed @ mode_observed)
            if denominator <= 0.0 or not np.isfinite(denominator):
                continue

            common_shock = float(
                mode_observed @ current_centered[current_observed] / denominator
            )
            residual = current_centered - mode * common_shock
            result[t, active_idx[current_observed]] = residual[current_observed]

    return pd.DataFrame(result, index=ret.index, columns=ret.columns)


@register_factor
class SamuelsonEigenmodeReceiverSlippage(BaseFactor):
    """Ranked reversal of delayed stock responses to a sector common-shock mode."""

    factor_id = "samuelson_eigenmode_receiver_slippage"
    name = "Eigenmode Receiver Slippage"
    category = "statistical_arbitrage"
    description = (
        "Using rolling 60-bar within-sector return covariance, the signal "
        "estimates each stock's loading on the leading sector eigenmode and "
        "measures its current return minus the loading-implied common-shock "
        "return. It cross-sectionally ranks the negative of that residual, "
        "with a gate requiring an elevated leading-eigenvalue share to ensure "
        "a genuine common-shock regime."
    )
    window_length = 60
    inputs = ["close", "sector"]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].astype(float)
        safe_close = close.replace(0.0, np.nan)
        ret = (safe_close / safe_close.shift(1) - 1.0).replace(
            [np.inf, -np.inf], np.nan
        )

        residual = _pre_shock_eigenmode_residual(ret, data["sector"], 60)
        signal = rank(-residual)
        return signal.fillna(0.0)
