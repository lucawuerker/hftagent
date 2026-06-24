"""Tracks A/B — signal-based factor analytics (LLM-free, near-free).

Two cheap analyses that reuse the factor signals already cached by the usability
probe (``modeling.service._SIGNAL_CACHE``), so they add almost no cost on top of
the IC track:

A. **diversity & redundancy** (:func:`evaluate_prerun_diversity`) — how much
   *unique* information a research zoo holds.  From the pairwise signal correlation
   matrix we report each factor's mean ``|corr|`` to the rest, the zoo's mean
   ``|corr|``, the **effective number of independent factors** (participation ratio
   of the correlation eigenvalues) and a **redundancy-cluster count** at ``|corr| ≥
   τ``.  This distinguishes a zoo of genuinely diverse alphas from one of near
   duplicate variants — something neither raw IC nor the brute-force ML track shows.

B. **deflation & model-based importance** (:func:`evaluate_prerun_modelview`) — IC
   corrected for selection, and which factors a model actually leans on.  The zoo's
   best ``|IC|`` is haircut for the number of factors tried (a bigger zoo has more
   lucky maxima — the expected max of ``k`` null factors grows like
   ``sqrt(2·ln k)``); a LASSO + gradient-boosting fit on the factors → forward
   return reports the top factors by weight/importance and how many factors carry
   non-zero weight (model-view redundancy).

Both build one pooled ``(timestamp×ticker) × factor`` feature matrix from the
cached signals and subsample to ``cfg.analytics_max_rows`` rows, so the cost is a
few seconds even on the intraday panel.
"""

from __future__ import annotations

import logging
import math
from typing import Any

log = logging.getLogger("comparison.analytics")


# ── shared: pooled feature matrix from the cached signals ────────────────────

def _feature_matrix(factor_ids: list[str], panel: dict[str, Any], cfg,
                    with_target: bool = False):
    """Pooled ``(obs × factors)`` matrix (+ optional forward-return target).

    Every factor's *cached*, standardised signal is reindexed to the shared
    ``close`` grid and flattened in the same row-major order, so column ``j`` of
    every factor lines up by ``(timestamp, ticker)``.  Standardisation follows
    ``cfg.fit_standardize``: ``per_underlying`` (the default) z-scores each factor
    per column **over time** — non-cross-sectional, so diversity + importance are
    well defined for a single ticker (and pool naturally across many);
    ``cross_sectional`` keeps the legacy across-tickers z-score.  Rows are then
    subsampled to ``cfg.analytics_max_rows`` (seeded, sorted to keep time order).
    """
    import numpy as np
    import pandas as pd

    from quant_fund_agent.backtesting.data_loader import forward_returns
    from quant_fund_agent.backtesting.strategy_backtester import normalise_factor_signals
    from quant_fund_agent.comparison.standardize import per_underlying_zscore
    from quant_fund_agent.modeling.service import _factor_signal

    close = panel["close"]
    ref_index, ref_cols = close.index, close.columns

    def _ravel(df) -> np.ndarray:
        return df.reindex(index=ref_index, columns=ref_cols).to_numpy(dtype=float).ravel()

    signals = {fid: _factor_signal(fid, panel) for fid in factor_ids}
    if getattr(cfg, "fit_standardize", "per_underlying") == "cross_sectional":
        signals = normalise_factor_signals(signals)  # across-tickers z-score
    else:  # per-underlying time-series z-score (default; works for one ticker)
        signals = {fid: per_underlying_zscore(s) for fid, s in signals.items()}
    cols = {fid: _ravel(signals[fid]) for fid in factor_ids}
    X = pd.DataFrame(cols, columns=list(factor_ids))
    y = pd.Series(_ravel(forward_returns(close, horizon=cfg.target_horizon))) if with_target else None

    if len(X) > cfg.analytics_max_rows:
        rng = np.random.default_rng(cfg.seed)
        idx = np.sort(rng.choice(len(X), size=cfg.analytics_max_rows, replace=False))
        X = X.iloc[idx].reset_index(drop=True)
        if y is not None:
            y = y.iloc[idx].reset_index(drop=True)
    return X, y


def _r(v: Any, nd: int = 6) -> float | None:
    try:
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else round(f, nd)
    except (TypeError, ValueError):
        return None


# ── Track A: diversity & redundancy ──────────────────────────────────────────

def _count_clusters(abs_corr, tau: float) -> int:
    """Connected-component count where factors i,j share an edge if |corr| ≥ τ."""
    import numpy as np

    adj = np.nan_to_num(abs_corr, nan=0.0) >= tau
    n = adj.shape[0]
    seen = [False] * n
    clusters = 0
    for i in range(n):
        if seen[i]:
            continue
        clusters += 1
        stack = [i]
        seen[i] = True
        while stack:
            u = stack.pop()
            for v in np.nonzero(adj[u])[0]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
    return clusters


def evaluate_prerun_diversity(
    prerun: str, factor_ids: list[str], panel: dict[str, Any], cfg,
    names: dict[str, str] | None = None,
):
    """Diversity/redundancy for one prerun → ``(summary_row, factor_rows, corr_df)``."""
    import numpy as np
    import pandas as pd

    names = names or {}
    n = len(factor_ids)
    if n < 2:
        summary = {"prerun": prerun, "n_factors": n, "mean_abs_corr": None,
                   "eff_n_factors": float(n), "eff_ratio": (1.0 if n else None),
                   "n_clusters": n, "redundancy": (0.0 if n else None)}
        return summary, [], pd.DataFrame()

    log.info("diversity: prerun '%s' — correlating %d factors", prerun, n)
    X, _ = _feature_matrix(factor_ids, panel, cfg)
    corr = X.corr()  # pairwise-complete Pearson

    C = corr.to_numpy(dtype=float).copy()  # writable (to_numpy may be read-only)
    np.fill_diagonal(C, 1.0)
    C = np.nan_to_num(C, nan=0.0)
    C = (C + C.T) / 2.0  # enforce symmetry for the eigen-decomposition

    # Effective number of independent factors = participation ratio of the
    # correlation eigenvalues: (Σλ)² / Σλ².  n for an orthogonal set, →1 if every
    # factor is the same direction.
    eig = np.clip(np.linalg.eigvalsh(C), 0.0, None)
    denom = float((eig ** 2).sum())
    eff = float(eig.sum() ** 2 / denom) if denom > 0 else float(n)
    eff = min(eff, float(n))

    abs_off = np.abs(C).copy()
    np.fill_diagonal(abs_off, np.nan)
    n_clusters = _count_clusters(abs_off, cfg.corr_threshold)

    import warnings
    with warnings.catch_warnings():  # all-NaN rows (constant factors) → NaN, not a crash
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_abs = np.nanmean(abs_off, axis=1)
        row_max = [np.nanmax(abs_off[i]) if np.isfinite(abs_off[i]).any() else np.nan
                   for i in range(n)]
        zoo_mean_abs = float(np.nanmean(mean_abs))

    factor_rows = [{
        "prerun": prerun, "factor_id": fid, "name": names.get(fid, fid),
        "mean_abs_corr": _r(mean_abs[i]),
        "max_abs_corr": _r(row_max[i]),
    } for i, fid in enumerate(factor_ids)]

    summary = {
        "prerun": prerun, "n_factors": n,
        "mean_abs_corr": _r(zoo_mean_abs),
        "eff_n_factors": _r(eff),
        "eff_ratio": _r(eff / n),
        "n_clusters": int(n_clusters),
        "redundancy": _r(1.0 - eff / n),  # 0 = fully independent, →1 = redundant
    }
    return summary, factor_rows, corr


# ── Track B: deflation + model-based importance ──────────────────────────────

def _deflate_ic(prerun: str, factor_ids: list[str], ic_rows, cfg) -> dict[str, Any]:
    """Haircut the zoo's best |IC| for the number of factors tried.

    The expected maximum of ``k`` independent null IC t-stats is ≈ ``sqrt(2·ln k)``;
    converted to an IC scale that is ``sqrt(2·ln k / n_obs)``.  ``deflated_best_ic``
    is the best ``|IC|`` minus that luck term (floored at 0); ``deflated_best_t``
    is the best IC t-stat minus ``sqrt(2·ln k)`` (>0 ⇒ beats selection luck).
    """
    h = cfg.target_horizon
    fids = set(factor_ids)
    rows = [r for r in (ic_rows or []) if r.get("prerun") == prerun and r.get("factor_id") in fids]
    ics = [abs(r[f"ic_{h}"]) for r in rows if r.get(f"ic_{h}") is not None]
    n_obs = max((int(r.get("n_timestamps") or 0) for r in rows), default=0)
    k = max(1, len(ics))
    if not ics or n_obs <= 1:
        return {"best_ic": (_r(max(ics)) if ics else None), "deflated_best_ic": None,
                "deflated_best_t": None, "ic_n_obs": n_obs, "ic_n_tested": k}

    best_ic = max(ics)
    expected_luck_ic = math.sqrt(2.0 * math.log(k)) / math.sqrt(n_obs) if k > 1 else 0.0
    best_t = best_ic * math.sqrt(n_obs)
    return {
        "best_ic": _r(best_ic),
        "deflated_best_ic": _r(max(best_ic - expected_luck_ic, 0.0)),
        "deflated_best_t": _r(best_t - (math.sqrt(2.0 * math.log(k)) if k > 1 else 0.0)),
        "ic_n_obs": n_obs, "ic_n_tested": k,
    }


def _fit_importance(model: str, X, y, cfg):
    """Fit one model on the pooled rows; return (normalised importances, n_nonzero)."""
    import numpy as np

    from quant_fund_agent.modeling.catalog import build_estimator, unwrap_estimator

    est = build_estimator(model, cfg.fast_model_params(model))  # scaler+model wrapped
    est.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
    inner = unwrap_estimator(est)
    if hasattr(inner, "feature_importances_"):
        imp = np.abs(np.asarray(inner.feature_importances_, dtype=float)).ravel()
    elif hasattr(inner, "coef_"):
        imp = np.abs(np.asarray(inner.coef_, dtype=float)).ravel()
    else:
        imp = np.zeros(X.shape[1])
    n_nonzero = int((imp > 1e-12).sum())
    total = imp.sum()
    if total > 0:
        imp = imp / total
    return imp, n_nonzero


def evaluate_prerun_modelview(
    prerun: str, factor_ids: list[str], panel: dict[str, Any], cfg,
    ic_rows=None, names: dict[str, str] | None = None, top_n: int = 10,
):
    """Deflation + model importance for one prerun → ``(summary_row, importance_rows)``."""
    import numpy as np

    names = names or {}
    summary: dict[str, Any] = {"prerun": prerun, "n_factors": len(factor_ids)}
    summary.update(_deflate_ic(prerun, factor_ids, ic_rows, cfg))

    importance_rows: list[dict[str, Any]] = []
    if not factor_ids:
        return summary, importance_rows

    X, y = _feature_matrix(factor_ids, panel, cfg, with_target=True)
    # Require only a valid target; impute missing factor values with 0 — the
    # cross-sectional mean after z-scoring.  (Requiring *every* one of 100+
    # heterogeneous microstructure factors to be non-NaN in a row leaves nothing.)
    ymask = y.notna()
    Xv, yv = X[ymask].fillna(0.0), y[ymask]
    if len(Xv) < 50:
        log.warning("modelview '%s': only %d rows with a valid target — skipping importance",
                    prerun, len(Xv))
        return summary, importance_rows

    log.info("modelview: prerun '%s' — importance on %d rows × %d factors (%s)",
             prerun, len(Xv), len(factor_ids), ",".join(cfg.importance_models))
    for model in cfg.importance_models:
        try:
            imp, n_nonzero = _fit_importance(model, Xv, yv, cfg)
        except Exception as e:  # noqa: BLE001 — one model must not abort the track
            log.warning("importance %s/%s failed: %s", prerun, model, e)
            continue
        if model == "lasso":
            summary["lasso_n_nonzero"] = int(n_nonzero)
            summary["lasso_sparsity"] = _r(1.0 - n_nonzero / len(factor_ids))
        for rank, j in enumerate(np.argsort(imp)[::-1][:top_n], start=1):
            importance_rows.append({
                "prerun": prerun, "model": model,
                "factor_id": factor_ids[j], "name": names.get(factor_ids[j], factor_ids[j]),
                "importance": _r(float(imp[j])), "rank": rank,
            })
    return summary, importance_rows
