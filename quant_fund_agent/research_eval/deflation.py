"""N_trials-aware deflation: the honesty layer of the evolutionary search.

The controller evaluates thousands of candidates and keeps the best out-of-sample
score.  Under the null (no real edge) the *maximum* of many noisy estimates is
large purely by luck — the expected max of ``k`` independent standard normals grows
like ``sqrt(2·ln k)``.  Selecting on it without correction is textbook
data-snooping.  These helpers inflate the significance bar by the number of trials
the search has run, wrapping the same haircut idea already used in
``comparison/analytics.py`` but keyed on ``N_trials`` (the *search* count) rather
than just the size of one zoo.

* :func:`deflated_ic` — haircut the best ``|IC|`` and its t-stat for ``N_trials``.
* :func:`expected_max_sharpe` / :func:`deflated_sharpe_ratio` — the Bailey &
  López de Prado (2014) Deflated Sharpe Ratio: the probability the observed Sharpe
  exceeds the expected maximum Sharpe under the null, adjusting for trial count,
  sample length and the return distribution's skew/kurtosis.

References: Bailey & López de Prado, *The Deflated Sharpe Ratio* (2014); Harvey,
Liu & Zhu, *…and the Cross-Section of Expected Returns* (2016).
"""

from __future__ import annotations

import math
from typing import Any

# Euler–Mascheroni constant, used in the expected-maximum-of-Gaussians term.
_EULER_GAMMA = 0.5772156649015329


# ── standard-normal helpers (scipy if present, else closed-form fallbacks) ────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation; scipy if available)."""
    try:  # scipy is available in this env, but keep a dependency-free fallback.
        from scipy.stats import norm  # type: ignore

        return float(norm.ppf(p))
    except Exception:  # noqa: BLE001
        pass
    if not 0.0 < p < 1.0:
        return -math.inf if p <= 0.0 else math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# ── IC deflation (multiple-testing haircut) ──────────────────────────────────

def ic_haircut(n_obs: int, n_trials: int) -> float:
    """The IC "luck" term: ``sqrt(2·ln N_trials / n_obs)`` (0 for a single trial).

    The expected maximum t-stat of ``N_trials`` independent null candidates is
    ``≈ sqrt(2·ln N_trials)``; dividing by ``sqrt(n_obs)`` converts it to the IC
    scale (since ``t ≈ IC·sqrt(n_obs)``).
    """
    if n_trials <= 1 or n_obs <= 1:
        return 0.0
    return math.sqrt(2.0 * math.log(n_trials)) / math.sqrt(n_obs)


def deflated_ic(best_abs_ic: float, n_obs: int, n_trials: int) -> dict[str, Any]:
    """Deflate the best ``|IC|`` observed over a search for the trial count.

    Returns ``deflated_ic`` (``|IC|`` minus the luck term, floored at 0),
    ``deflated_t`` (the IC t-stat minus ``sqrt(2·ln N_trials)`` — ``> 0`` means the
    edge beats selection luck), plus the components for transparency.
    """
    if best_abs_ic is None or n_obs <= 1:
        return {"best_ic": best_abs_ic, "deflated_ic": None, "deflated_t": None,
                "luck_ic": None, "n_obs": int(n_obs), "n_trials": int(n_trials)}
    best_abs_ic = abs(float(best_abs_ic))
    luck_ic = ic_haircut(n_obs, n_trials)
    best_t = best_abs_ic * math.sqrt(n_obs)
    luck_t = math.sqrt(2.0 * math.log(n_trials)) if n_trials > 1 else 0.0
    return {
        "best_ic": round(best_abs_ic, 6),
        "deflated_ic": round(max(best_abs_ic - luck_ic, 0.0), 6),
        "deflated_t": round(best_t - luck_t, 6),
        "luck_ic": round(luck_ic, 6),
        "n_obs": int(n_obs),
        "n_trials": int(n_trials),
    }


# ── Deflated Sharpe Ratio (Bailey & López de Prado, 2014) ─────────────────────

def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """Expected maximum Sharpe under the null across ``n_trials`` independent trials.

    ``SR0 = sqrt(Var[SR]) · [(1-γ)·Z⁻¹(1 - 1/N) + γ·Z⁻¹(1 - 1/(N·e))]`` where ``γ``
    is the Euler–Mascheroni constant and ``Z⁻¹`` the inverse standard-normal CDF.
    ``sr_variance`` is the variance of the Sharpe estimates *across the trials* (the
    dispersion of the strategies searched).  A single trial has no selection bias so
    ``SR0 = 0``.
    """
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    n = float(n_trials)
    term = ((1.0 - _EULER_GAMMA) * _norm_ppf(1.0 - 1.0 / n)
            + _EULER_GAMMA * _norm_ppf(1.0 - 1.0 / (n * math.e)))
    return math.sqrt(sr_variance) * term


def deflated_sharpe_ratio(
    observed_sr: float,
    *,
    n_obs: int,
    sr_star: float | None = None,
    sr_variance: float | None = None,
    n_trials: int | None = None,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float | None:
    """Probability the observed Sharpe beats the expected max Sharpe under the null.

    ``observed_sr`` and the threshold ``sr_star`` must be in the *same* (per-bar,
    non-annualised) units.  Provide ``sr_star`` directly, or let it be computed via
    :func:`expected_max_sharpe` from ``sr_variance`` + ``n_trials``.  ``skew`` and
    ``kurtosis`` (non-excess; 3 = Gaussian) correct for non-normal returns.  Returns
    a probability in ``[0, 1]`` (higher = more likely a genuine edge), or ``None`` if
    it cannot be computed (too few observations, or a degenerate denominator).
    """
    if n_obs is None or n_obs <= 1 or observed_sr is None:
        return None
    if sr_star is None:
        if sr_variance is None or n_trials is None:
            raise ValueError("Provide sr_star, or both sr_variance and n_trials.")
        sr_star = expected_max_sharpe(sr_variance, n_trials)

    denom_var = 1.0 - skew * observed_sr + (kurtosis - 1.0) / 4.0 * observed_sr ** 2
    if denom_var <= 0:
        return None
    z = (observed_sr - sr_star) * math.sqrt(n_obs - 1) / math.sqrt(denom_var)
    return _norm_cdf(z)


# ── Probability of Backtest Overfitting (Bailey et al., 2016; CSCV) ───────────

def pbo_cscv(performance: Any, n_groups: int = 8) -> dict[str, Any]:
    """Probability of Backtest Overfitting via Combinatorially Symmetric CV.

    ``performance`` is a (T × N) array-like: T time observations of a
    per-period performance metric (e.g. returns of the combined signal) for N
    candidate configurations — here, N factors/strategies the search evaluated.
    The T rows are cut into ``n_groups`` contiguous blocks; for every balanced
    split of the blocks into train/test halves, the candidate with the best
    *train* Sharpe is located in the *test* ranking.  ``PBO`` is the fraction
    of splits where that in-sample winner falls in the bottom half
    out-of-sample — the probability that selecting the backtest winner selects
    an overfit strategy.

    Returns ``{"pbo", "n_splits", "logits"}`` (``pbo=None`` when undefined:
    fewer than 2 candidates or too little data).  Headline thesis metric,
    reported alongside the deflated Sharpe.
    """
    import itertools

    import numpy as np

    M = np.asarray(performance, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2 or M.shape[0] < n_groups or n_groups < 2 \
            or n_groups % 2 != 0:
        return {"pbo": None, "n_splits": 0, "logits": []}

    blocks = np.array_split(np.arange(M.shape[0]), n_groups)

    def _sharpe(rows: np.ndarray) -> np.ndarray:
        sub = M[rows]
        mu = np.nanmean(sub, axis=0)
        sd = np.nanstd(sub, axis=0)
        sd[sd == 0] = np.nan
        return mu / sd

    logits: list[float] = []
    n = M.shape[1]
    for combo in itertools.combinations(range(n_groups), n_groups // 2):
        train_rows = np.concatenate([blocks[i] for i in combo])
        test_rows = np.concatenate([blocks[i] for i in range(n_groups)
                                    if i not in combo])
        sr_train = _sharpe(train_rows)
        sr_test = _sharpe(test_rows)
        if np.all(~np.isfinite(sr_train)) or np.all(~np.isfinite(sr_test)):
            continue
        best = int(np.nanargmax(sr_train))
        # relative rank of the IS winner in the OOS ordering, ω ∈ (0, 1)
        omega = (np.sum(sr_test <= sr_test[best]) - 0.5) / n
        omega = min(max(omega, 1e-9), 1 - 1e-9)
        logits.append(math.log(omega / (1.0 - omega)))

    if not logits:
        return {"pbo": None, "n_splits": 0, "logits": []}
    pbo = float(np.mean([l <= 0 for l in logits]))
    return {"pbo": pbo, "n_splits": len(logits),
            "logits": [round(l, 6) for l in logits]}
