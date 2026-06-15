"""Portfolio construction primitives.

Each function returns a ``PortfolioWeights`` mapping ``strategy_id ->
weight``, with ``sum(|w|) == 1`` (long-only methods additionally
guarantee ``sum(w) == 1`` and ``w_i >= 0``).

Why these specific methods?
---------------------------
The PM agent is conceptually a *selector* over construction methods.
We cover the main families you would see in a real fund:

* **equal_weight**          – the 1/N benchmark.  Always available, never
  fails, very hard to beat without good covariance estimates.
* **inverse_volatility**    – cheap risk-balancing without a covariance
  matrix; useful when correlations are noisy.
* **min_variance**          – uses the covariance matrix only (no return
  forecasts needed) — robust because return forecasts are noisy.
* **mean_variance**         – classical Markowitz with a configurable
  risk-aversion λ.
* **max_sharpe**            – tangency portfolio against the risk-free
  rate (special case of mean-variance).
* **risk_parity**           – equal-risk-contribution; popular in
  multi-strategy funds.
* **hierarchical_risk_parity** – Lopez de Prado (2016).  Robust to
  ill-conditioned covariance matrices, which is common when the number
  of strategies is comparable to the length of their return histories.

All solvers use ``scipy.optimize`` (already a dependency) so we do *not*
introduce ``cvxpy``.  For long-only, fully-invested portfolios with N
strategies the QPs are tiny (≤ a few dozen) and SLSQP handles them in
microseconds.
"""

from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage

from quant_fund_agent.schemas import ConstructionMethod

log = logging.getLogger("portfolio.construction")

# Type alias — readability only.
PortfolioWeights = dict[str, float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _align_cov(strategy_ids: list[str], cov: pd.DataFrame) -> pd.DataFrame:
    """Reorder/restrict the covariance matrix to the requested IDs.

    Missing rows/columns — and any NaN cells, whether from an
    insufficient-overlap pair (``DataFrame.cov(min_periods=…)`` returns NaN
    there) or a strict-JSON round-trip over MCP (NaN→null→None, object
    dtype) — are filled with zero off-diagonal and the median variance on
    the diagonal so the matrix stays PSD and float-typed.  This is only a
    safety net — callers should pass a properly-sized cov.
    """
    # A strict-JSON round-trip turns NaN into None and the frame into
    # object dtype; coerce back to float so a None can never reach the
    # matmul as ``None * float`` (it becomes NaN, handled just below).
    cov = cov.apply(pd.to_numeric, errors="coerce")

    missing = [s for s in strategy_ids if s not in cov.index]
    if missing:
        log.warning("Covariance missing rows for %s — filling with median variance.", missing)

    # Median of the finite diagonal is the fallback variance for any
    # missing/NaN diagonal entry (1e-4 if the whole diagonal is unusable).
    diag = np.diag(cov.values).astype(float) if len(cov) else np.array([])
    finite = diag[np.isfinite(diag)]
    med_var = float(np.median(finite)) if finite.size else 1e-4

    aligned = cov.reindex(index=strategy_ids, columns=strategy_ids)
    vals = aligned.to_numpy(dtype=float, copy=True)
    n = len(strategy_ids)
    for i in range(n):
        if not np.isfinite(vals[i, i]):
            vals[i, i] = med_var
        for j in range(n):
            if i != j and not np.isfinite(vals[i, j]):
                vals[i, j] = 0.0
    return pd.DataFrame(vals, index=strategy_ids, columns=strategy_ids)


def _normalise(w: np.ndarray) -> np.ndarray:
    """L1-renormalise (||w||_1 = 1).  Returns zeros if w is all-zero."""
    s = np.abs(w).sum()
    if s == 0:
        return w
    return w / s


def _zip_weights(strategy_ids: list[str], w: np.ndarray) -> PortfolioWeights:
    return {sid: float(w[i]) for i, sid in enumerate(strategy_ids)}


# ---------------------------------------------------------------------------
# 1) Equal weight
# ---------------------------------------------------------------------------

def equal_weight(strategy_ids: list[str]) -> PortfolioWeights:
    """1/N over the given strategies.  Always returns valid weights."""
    if not strategy_ids:
        return {}
    w = 1.0 / len(strategy_ids)
    return {sid: w for sid in strategy_ids}


# ---------------------------------------------------------------------------
# 2) Inverse volatility
# ---------------------------------------------------------------------------

def inverse_volatility(vols: Mapping[str, float]) -> PortfolioWeights:
    """w_i ∝ 1 / σ_i, then renormalise.  Long-only, sums to 1.

    Strategies with σ <= 0 or non-finite σ are dropped.
    """
    cleaned = {k: float(v) for k, v in vols.items() if v and np.isfinite(v) and v > 0}
    if not cleaned:
        return {}
    raw = np.array([1.0 / cleaned[k] for k in cleaned])
    w = raw / raw.sum()
    return {sid: float(w[i]) for i, sid in enumerate(cleaned.keys())}


# ---------------------------------------------------------------------------
# 3) Minimum variance  (long-only, fully invested)
# ---------------------------------------------------------------------------

def min_variance(
    strategy_ids: list[str],
    cov: pd.DataFrame,
    allow_short: bool = False,
    max_weight: float | None = None,
) -> PortfolioWeights:
    """min_w  w' Σ w   s.t.  sum(w) = 1, w_i ∈ [lb, ub]."""
    if not strategy_ids:
        return {}
    cov = _align_cov(strategy_ids, cov.copy())
    Sigma = cov.values
    n = len(strategy_ids)

    x0 = np.full(n, 1.0 / n)
    lb = -1.0 if allow_short else 0.0
    ub = max_weight if max_weight is not None else 1.0
    bounds = [(lb, ub)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    res = minimize(
        fun=lambda w: float(w @ Sigma @ w),
        x0=x0,
        jac=lambda w: 2.0 * Sigma @ w,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 200, "ftol": 1e-10},
    )
    if not res.success:
        log.warning("min_variance solver failed (%s); falling back to equal weight.", res.message)
        return equal_weight(strategy_ids)
    return _zip_weights(strategy_ids, res.x)


# ---------------------------------------------------------------------------
# 4) Mean-variance  (Markowitz with explicit risk aversion λ)
# ---------------------------------------------------------------------------

def mean_variance(
    expected_returns: Mapping[str, float],
    cov: pd.DataFrame,
    risk_aversion: float = 1.0,
    allow_short: bool = False,
    max_weight: float | None = None,
) -> PortfolioWeights:
    """max_w  μ' w  -  (λ/2) w' Σ w   s.t.  sum(w) = 1, bounds.

    A larger ``risk_aversion`` (λ) → more weight on variance, the
    portfolio behaves more like min-variance.  λ → 0 collapses to the
    return-maximising corner solution.
    """
    strategy_ids = list(expected_returns.keys())
    if not strategy_ids:
        return {}
    cov = _align_cov(strategy_ids, cov.copy())
    Sigma = cov.values
    mu = np.array([expected_returns[s] for s in strategy_ids])
    n = len(strategy_ids)

    x0 = np.full(n, 1.0 / n)
    lb = -1.0 if allow_short else 0.0
    ub = max_weight if max_weight is not None else 1.0
    bounds = [(lb, ub)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def neg_utility(w: np.ndarray) -> float:
        return float(-(mu @ w) + 0.5 * risk_aversion * (w @ Sigma @ w))

    def neg_utility_grad(w: np.ndarray) -> np.ndarray:
        return -mu + risk_aversion * (Sigma @ w)

    res = minimize(
        fun=neg_utility,
        x0=x0,
        jac=neg_utility_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 200, "ftol": 1e-10},
    )
    if not res.success:
        log.warning("mean_variance solver failed (%s); falling back to equal weight.", res.message)
        return equal_weight(strategy_ids)
    return _zip_weights(strategy_ids, res.x)


# ---------------------------------------------------------------------------
# 5) Max-Sharpe  (tangency portfolio)
# ---------------------------------------------------------------------------

def max_sharpe(
    expected_returns: Mapping[str, float],
    cov: pd.DataFrame,
    risk_free_rate: float = 0.0,
    allow_short: bool = False,
    max_weight: float | None = None,
) -> PortfolioWeights:
    """Tangency portfolio: max_w  (μ' w − r_f) / sqrt(w' Σ w).

    We **do not** optimise the Sharpe ratio directly — it is scale
    invariant in w, so its gradient is always orthogonal to the
    feasible radial direction and SLSQP's line search trips with
    "positive directional derivative" even when the iterate is
    optimal.  Instead we use:

    1.  ``allow_short=True``  → closed-form tangency.
        With only ``sum(w) = 1`` as a constraint, the unique solution
        is ``w* = Σ⁻¹a / (1ᵀ Σ⁻¹a)`` where ``a = μ − r_f·1``.  No
        numerical solver needed; always correct when ``Σ`` is invertible.

    2.  ``allow_short=False`` → Schaefer (1980) transformation.
        Substitute ``y = w/(aᵀw)``.  Then the original problem is
        equivalent to the convex QP::

            min   yᵀ Σ y
            s.t.  aᵀ y = 1,
                  y ≥ 0,
                  (optional)  y_i ≤ ub · Σy   ∀i

        and we recover ``w = y / Σy``.  This is a plain QP — SLSQP
        solves it reliably.

    Both paths gracefully fall back when the inputs make max-Sharpe
    undefined (e.g. all expected excess returns ≤ 0).
    """
    strategy_ids = list(expected_returns.keys())
    if not strategy_ids:
        return {}

    cov = _align_cov(strategy_ids, cov.copy())
    Sigma = cov.values
    mu = np.array([expected_returns[s] for s in strategy_ids])
    a = mu - risk_free_rate
    n = len(strategy_ids)

    # ── Path 1: closed-form tangency for the shortable case ─────────
    if allow_short:
        try:
            Sigma_inv_a = np.linalg.solve(Sigma, a)
            denom = float(Sigma_inv_a.sum())
            if abs(denom) < 1e-12:
                raise np.linalg.LinAlgError("degenerate denominator")
            w = Sigma_inv_a / denom
            return _zip_weights(strategy_ids, w)
        except np.linalg.LinAlgError as e:
            log.warning("max_sharpe closed-form failed (%s); falling back to "
                        "long-only Schaefer.", e)
            # Fall through to the long-only QP as a safety net.

    # ── Path 2: long-only Schaefer transformation ───────────────────
    # Feasibility check: we need some long-only portfolio with
    # positive excess return, otherwise the tangency portfolio is
    # undefined (the efficient frontier doesn't intersect a > 0).
    if (a <= 0).all() or a.sum() <= 0:
        log.info("max_sharpe: no long-only portfolio has positive excess "
                 "return — falling back to equal weight.")
        return equal_weight(strategy_ids)

    # Starting point: y0 = (1/N · 1) scaled so aᵀy0 = 1.  Since
    # aᵀ(1/N · 1) = (Σa)/N, we set y0 = (N / Σa) · (1/N · 1) = 1/Σa.
    y0 = np.full(n, 1.0 / a.sum())

    eq_cons = [
        {"type": "eq",
         "fun": lambda y: float(a @ y - 1.0),
         "jac": lambda y: a}
    ]

    ineq_cons: list[dict] = []
    if max_weight is not None and max_weight < 1.0:
        # Encode  w_i ≤ ub  ⇔  ub · Σy − y_i ≥ 0  (linear in y).
        for i in range(n):
            def _fn(y, i=i):
                return float(max_weight * y.sum() - y[i])
            def _jac(y, i=i):
                g = np.full(n, max_weight)
                g[i] -= 1.0
                return g
            ineq_cons.append({"type": "ineq", "fun": _fn, "jac": _jac})

    res = minimize(
        fun=lambda y: float(y @ Sigma @ y),
        jac=lambda y: 2.0 * (Sigma @ y),
        x0=y0,
        method="SLSQP",
        bounds=[(0.0, None)] * n,
        constraints=eq_cons + ineq_cons,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    if not res.success:
        log.warning("max_sharpe (Schaefer QP) failed (%s); falling back to "
                    "inverse-volatility.", res.message)
        vols = {s: float(np.sqrt(Sigma[i, i])) for i, s in enumerate(strategy_ids)}
        return inverse_volatility(vols)

    y = res.x
    y_sum = float(y.sum())
    if y_sum <= 1e-12:
        log.warning("max_sharpe (Schaefer QP) collapsed (Σy ≈ 0); "
                    "falling back to equal weight.")
        return equal_weight(strategy_ids)

    w = y / y_sum
    return _zip_weights(strategy_ids, w)


# ---------------------------------------------------------------------------
# 6) Risk parity  (equal risk contribution)
# ---------------------------------------------------------------------------

def risk_parity(
    strategy_ids: list[str],
    cov: pd.DataFrame,
    max_iter: int = 500,
) -> PortfolioWeights:
    """Find long-only w with sum(w)=1 such that each strategy contributes
    equally to total portfolio variance.

    Solves  min_w  Σ_i (RC_i − 1/N)^2  with  RC_i = w_i (Σw)_i / (w'Σw).
    """
    if not strategy_ids:
        return {}
    cov = _align_cov(strategy_ids, cov.copy())
    Sigma = cov.values
    n = len(strategy_ids)
    target = 1.0 / n

    x0 = np.full(n, 1.0 / n)
    bounds = [(1e-6, 1.0)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def loss(w: np.ndarray) -> float:
        port_var = float(w @ Sigma @ w)
        if port_var <= 1e-12:
            return 1e6
        rc = w * (Sigma @ w) / port_var
        return float(np.sum((rc - target) ** 2))

    res = minimize(
        fun=loss,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": max_iter, "ftol": 1e-12},
    )
    if not res.success:
        log.warning("risk_parity solver failed (%s); falling back to inverse-vol.", res.message)
        vols = {s: float(np.sqrt(cov.loc[s, s])) for s in strategy_ids}
        return inverse_volatility(vols)
    return _zip_weights(strategy_ids, res.x)


# ---------------------------------------------------------------------------
# 7) Hierarchical Risk Parity (Lopez de Prado, 2016)
# ---------------------------------------------------------------------------

def _corr_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """De Prado's correlation distance: d_ij = sqrt(0.5 * (1 - ρ_ij))."""
    arr = np.sqrt(0.5 * (1.0 - corr.clip(-1.0, 1.0).values))
    arr = np.array(arr, copy=True)
    np.fill_diagonal(arr, 0.0)
    return pd.DataFrame(arr, index=corr.index, columns=corr.columns)


def _quasi_diag(link: np.ndarray) -> list[int]:
    """Walk the linkage matrix to obtain the quasi-diagonalisation order."""
    link = link.astype(int)
    n_items = link[-1, 3]
    sort_ix = [int(link[-1, 0]), int(link[-1, 1])]
    while max(sort_ix) >= n_items:
        new_sort = []
        for ix in sort_ix:
            if ix < n_items:
                new_sort.append(ix)
            else:
                row = link[ix - n_items]
                new_sort.extend([int(row[0]), int(row[1])])
        sort_ix = new_sort
    return sort_ix


def _hrp_bisect(cov: pd.DataFrame, sort_ix: list[int]) -> pd.Series:
    """Recursive bisection step from de Prado (2016)."""
    w = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]
    while clusters:
        new_clusters = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            half = len(cl) // 2
            left, right = cl[:half], cl[half:]
            v_left = _inverse_var_cluster_variance(cov, left)
            v_right = _inverse_var_cluster_variance(cov, right)
            alpha = 1.0 - v_left / (v_left + v_right)
            w[left] *= alpha
            w[right] *= 1.0 - alpha
            new_clusters.extend([left, right])
        clusters = new_clusters
    return w


def _inverse_var_cluster_variance(cov: pd.DataFrame, indices: list[int]) -> float:
    sub = cov.iloc[indices, indices].values
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hierarchical_risk_parity(
    strategy_ids: list[str],
    cov: pd.DataFrame,
) -> PortfolioWeights:
    """HRP per Lopez de Prado (2016) — robust to ill-conditioned covariance.

    Works in three steps:
      1. Build the correlation distance matrix.
      2. Single-linkage clustering → quasi-diagonalise covariance.
      3. Recursive bisection allocating inversely to cluster variance.
    """
    if not strategy_ids:
        return {}
    if len(strategy_ids) == 1:
        return {strategy_ids[0]: 1.0}

    cov = _align_cov(strategy_ids, cov.copy())
    std = np.sqrt(np.diag(cov.values))
    corr_vals = cov.values / np.outer(std, std)
    corr = pd.DataFrame(corr_vals, index=cov.index, columns=cov.columns)

    d = _corr_distance(corr)
    condensed = squareform(d.values, checks=False)
    link = linkage(condensed, method="single")

    sort_ix = _quasi_diag(link)
    cov_reset = cov.copy()
    cov_reset.index = range(len(cov_reset))
    cov_reset.columns = range(len(cov_reset))
    w_sorted = _hrp_bisect(cov_reset, sort_ix)

    # Map positional indices back to strategy IDs.
    weights = {strategy_ids[i]: float(w_sorted[i]) for i in w_sorted.index}
    # Renormalise to be safe against rounding.
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights


# ---------------------------------------------------------------------------
# Expected metrics (used to populate PortfolioRecord.expected_metrics)
# ---------------------------------------------------------------------------

def expected_portfolio_metrics(
    weights: PortfolioWeights,
    expected_returns: Mapping[str, float] | None,
    cov: pd.DataFrame,
) -> dict[str, float]:
    """Compute simple ex-ante portfolio statistics from weights + cov.

    All quantities are in whatever units ``expected_returns`` and ``cov``
    are in — the caller decides whether they're annualised or per-bar.
    """
    sids = list(weights.keys())
    if not sids:
        return {}
    cov = _align_cov(sids, cov.copy())
    w = np.array([weights[s] for s in sids])
    port_var = float(w @ cov.values @ w)
    port_vol = float(np.sqrt(max(port_var, 0.0)))
    out: dict[str, float] = {
        "expected_volatility": port_vol,
        "expected_variance": port_var,
        "gross_exposure": float(np.abs(w).sum()),
        "net_exposure": float(w.sum()),
    }
    if expected_returns:
        mu = np.array([expected_returns.get(s, 0.0) for s in sids])
        port_ret = float(mu @ w)
        out["expected_return"] = port_ret
        out["expected_sharpe"] = port_ret / port_vol if port_vol > 0 else 0.0
    return out


# ---------------------------------------------------------------------------
# Dispatcher used by the PM agent
# ---------------------------------------------------------------------------

def construct_portfolio(
    method: ConstructionMethod,
    strategy_ids: list[str],
    cov: pd.DataFrame | None = None,
    expected_returns: Mapping[str, float] | None = None,
    *,
    risk_aversion: float = 1.0,
    risk_free_rate: float = 0.0,
    allow_short: bool = False,
    max_weight: float | None = None,
) -> PortfolioWeights:
    """Pick a construction method by enum and route to the right function.

    This is the single entry point the PM agent calls.
    """
    if not strategy_ids:
        return {}

    # Feasibility clamp: long-only fully-invested optimisers need
    # ``max_weight * N >= 1``.  If the personality is more restrictive
    # than that we relax the per-strategy cap to ``1/N`` so the QP has a
    # feasible region.  Without this clamp SLSQP silently fails and we
    # fall back to equal weight.
    if max_weight is not None and not allow_short:
        min_feasible = 1.0 / len(strategy_ids)
        if max_weight < min_feasible:
            log.info(
                "Relaxing max_weight from %.3f to %.3f (1/N) to keep "
                "the constraint set feasible.",
                max_weight, min_feasible,
            )
            max_weight = min_feasible

    if method == ConstructionMethod.EQUAL_WEIGHT:
        return equal_weight(strategy_ids)

    if method == ConstructionMethod.INVERSE_VOLATILITY:
        if cov is None:
            raise ValueError("inverse_volatility requires a covariance matrix")
        vols = {s: float(np.sqrt(cov.loc[s, s])) for s in strategy_ids if s in cov.index}
        return inverse_volatility(vols)

    if cov is None:
        raise ValueError(f"{method.value} requires a covariance matrix")

    if method == ConstructionMethod.MIN_VARIANCE:
        return min_variance(strategy_ids, cov,
                            allow_short=allow_short, max_weight=max_weight)

    if method == ConstructionMethod.MEAN_VARIANCE:
        if expected_returns is None:
            raise ValueError("mean_variance requires expected_returns")
        return mean_variance(
            {s: expected_returns.get(s, 0.0) for s in strategy_ids},
            cov,
            risk_aversion=risk_aversion,
            allow_short=allow_short,
            max_weight=max_weight,
        )

    if method == ConstructionMethod.MAX_SHARPE:
        if expected_returns is None:
            raise ValueError("max_sharpe requires expected_returns")
        return max_sharpe(
            {s: expected_returns.get(s, 0.0) for s in strategy_ids},
            cov,
            risk_free_rate=risk_free_rate,
            allow_short=allow_short,
            max_weight=max_weight,
        )

    if method == ConstructionMethod.RISK_PARITY:
        return risk_parity(strategy_ids, cov)

    if method == ConstructionMethod.HIERARCHICAL_RISK_PARITY:
        return hierarchical_risk_parity(strategy_ids, cov)

    if method == ConstructionMethod.CUSTOM_LLM:
        # The LLM produced weights directly; the PM agent supplies them.
        # Falling back to equal weight here is a safety net.
        return equal_weight(strategy_ids)

    raise ValueError(f"Unknown construction method: {method}")
