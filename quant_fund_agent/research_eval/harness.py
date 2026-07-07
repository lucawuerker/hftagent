"""Deterministic fitness harness — the un-gameable *reward* channel.

This turns one candidate factor **signal** (plus the signals of the current book /
Pareto archive) into a :class:`~quant_fund_agent.research_eval.fitness.FitnessResult`
— the CORE Pareto objective vector + the hard-gate booleans (the selection channel)
and a rich diagnostics dict (the teacher channel handed to Reflection / the LLM).

It is deliberately *signal-oriented* (it takes DataFrames, not factor ids or LLM
output), so it is fully testable on a synthetic panel and can never be influenced by
an LLM.  It is mostly **orchestration** of trusted code that already exists —
``comparison/standardize`` (per-underlying z-score), ``comparison/ic`` (pooled
Spearman IC), ``comparison/analytics`` (participation ratio), ``modeling/catalog``
(estimators), ``backtesting/data_loader`` (forward returns) — glued to the split /
deflation seams in this package.

Feedback families implemented (tags per ``docs/research-evolution/DESIGN.md``):

* Family 1 (standalone, ``[DIAG]``): IC at the forecast horizon + an IC-decay curve.
* Family 2 (marginal, ``[CORE, primary]``): LOCO ΔOOS-IC of the combined model from
  adding the candidate to the book; residual (orthogonalised) IC as a diagnostic.
* Family 3 (independence, ``[CORE]``): residual IC by default (legacy
  Δ participation ratio − soft max-|corr| is still selectable).
* Family 4 (robustness, ``[CORE]``): fold-refit CPCV combined/LOCO IC
  distribution, plus the OOS/IS degradation and deflation diagnostics.
* Family 5 (realism): coverage gate + hypothesis sign-consistency.

* Family 4 also includes the parameter-sensitivity **plateau penalty** when the
  caller supplies ``jitter_signals`` (signals of window-jittered variants of the
  candidate, produced by the P1 jitter mutation operator): a candidate whose VAL
  IC collapses under a ±10% window jitter sits on a knife-edge, not a plateau,
  and its robustness is docked by the IC drop.

Not yet wired (documented for later phases): the transaction-cost gate (P5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np
import pandas as pd

from quant_fund_agent.backtesting.data_loader import forward_returns
from quant_fund_agent.comparison.ic import _spearman
from quant_fund_agent.comparison.standardize import per_underlying_zscore
from quant_fund_agent.research_eval import deflation
from quant_fund_agent.research_eval.fitness import (
    FitnessResult,
    GateResults,
    ObjectiveVector,
    complexity,
    participation_ratio,
)
from quant_fund_agent.research_eval.splits import ThreeWaySplit, cpcv_folds, three_way_split

log = logging.getLogger("research_eval.harness")


@dataclass
class EvalParams:
    """Tunable knobs for the deterministic evaluation (all deterministic / seeded).

    These parameterise the *scoring*, never the search; they are held identical
    across every candidate in a run so the Pareto comparison is fair.
    """

    n_trials: int = 1                 # candidates evaluated so far (drives deflation)
    lambda_std: float = 1.0           # robustness penalty on CPCV IC dispersion
    corr_penalty: float = 0.5         # weight of the soft max-|corr| independence penalty
    sign_bonus: float = 0.02          # ± robustness bonus for hypothesis sign (mis)match
    # Estimator combining the book's factors for the LOCO marginal-value axis.
    # Default is a NONLINEAR model (gradient boosting) on purpose: a linear model
    # (ridge) only sees *additive* combination value and scores a pure
    # conditioning/state variable (the classic low-standalone-IC volatility factor,
    # valuable only via `vol × momentum`-style interactions) at ~0 marginal.  A
    # tree ensemble captures those interactions, so conditioning factors score a
    # positive marginal value.  Override for an ablation (`ridge` reproduces the
    # additive-only behaviour); any `modeling.catalog` model id is accepted.
    marginal_model: str = "gradient_boosting"
    plateau_weight: float = 1.0       # weight of the window-jitter plateau penalty
    cpcv_groups: int = 6
    cpcv_k: int = 2
    embargo: int = 0
    # Main robustness axis: refit a combined model on each CPCV fold's train mask
    # and score its OOS fold-test IC.  ``None`` reuses ``marginal_model``; a caller
    # can set e.g. ``lightgbm`` for faster nonlinear refits when available.
    cpcv_model: str | None = None
    cpcv_fast: bool = True             # use ComparisonConfig's lightweight tree params
    gate_coverage: float = 0.5        # τ_cov: min non-NaN (date,ticker) fraction
    gate_degradation: float = 0.5     # τ_deg: min OOS/IS IC ratio (same sign)
    min_is_ic: float = 0.005          # |IS IC| below this → degradation gate not evaluated
    ic_decay_horizons: tuple[int, ...] = (1, 3, 6, 12, 24)
    # ── independence axis basis ──
    # "residual_ic": the candidate's predictive edge orthogonal to the book (novel
    # content; the default — un-saturating, rewards independence that *predicts*).
    # "delta_participation": the legacy Δ-participation-ratio − max-|corr| penalty.
    independence_metric: str = "residual_ic"
    # ── regime (5th axis): reward edge in stress/crash bars where the book is weak ──
    regime_kind: str = "drawdown"     # "drawdown" (worst market-return bars) | "volatility"
    regime_quantile: float = 0.2      # tail fraction of dev bars labelled "stress"
    regime_vol_window: int = 20       # rolling window (bars) for the volatility regime
    regime_min_obs: int = 20          # min scored (date,ticker) cells for a valid regime IC
    # ── economic realism (P5, WS3) — folded in, NO new Pareto axis ──
    # cost_ok gate: reject a factor whose per-bar turnover exceeds this floor.  None
    # (default) = gate not evaluated, so the baseline arm is byte-identical; a value
    # (mean |Δposition| in [0, 2]) activates it.  Turnover / net-of-cost are always
    # computed as diagnostics regardless of the gate.
    gate_turnover: float | None = None
    cost_rate: float = 5e-4                     # ≈5 bps per unit turnover (net-of-cost DIAG)
    turnover_position_mode: str = "threshold"   # shared position primitive for the measure
    turnover_position_threshold: float = 1.0
    turnover_zscore_basis: str = "expanding"    # causal, leak-free per-underlying z-score
    # perturbation-fidelity probe (AlphaEval) folded into the robustness axis: dock
    # robustness by the sign-aligned VAL-IC drop under a Gaussian signal shock.  Weight
    # 0.0 (default) = off (baseline arm unchanged); > 0 activates the extra probe (one
    # more robustness term, like the plateau penalty — not a new axis).
    perturbation_weight: float = 0.0
    perturbation_sigma: float = 0.5            # shock stdev in signal z-units
    perturbation_draws: int = 3


# ── low-level IC / prediction helpers (signal-based, leak-aware) ──────────────

def _pooled_ic(
    sig: pd.DataFrame, close: pd.DataFrame, horizon: int,
    row_mask: np.ndarray | None = None, stat_mask: np.ndarray | None = None,
    available_mask: np.ndarray | None = None,
) -> tuple[float | None, int]:
    """Pooled per-underlying time-series Spearman IC of ``sig`` vs its forward return.

    ``row_mask`` selects the rows scored; ``stat_mask`` selects the rows whose stats
    standardise the signal (pass the IS window to keep an OOS score leak-free).  The
    forward return is taken from ``close`` after optionally applying
    ``available_mask``: if a row's ``t+horizon`` price is outside the available
    development window, that row is dropped so a validation score can never consume
    held-out TEST prices.
    """
    s = sig.reindex(index=close.index, columns=close.columns)
    stat_idx = close.index[stat_mask] if stat_mask is not None else None
    z = per_underlying_zscore(s, stat_idx)
    x = z.to_numpy(dtype=float)
    y = forward_returns(close, horizon=horizon).to_numpy(dtype=float)
    if available_mask is not None:
        valid = _label_available_mask(available_mask, horizon)
        row_mask = valid if row_mask is None else (np.asarray(row_mask) & valid)
    if row_mask is not None:
        x, y = x[row_mask], y[row_mask]
    return _spearman(x.ravel(), y.ravel())


def _combined_prediction(
    signals: Sequence[pd.DataFrame], close: pd.DataFrame, is_mask: np.ndarray,
    cfg: Any, model: str,
) -> pd.DataFrame | None:
    """Fit ``model`` on the book's standardised signals (IS rows) → (T×N) prediction.

    Mirrors ``comparison.bruteforce`` but consumes signal frames directly.  Features
    are standardised per-underlying using IS-window stats (or cross-sectionally when
    ``cfg.fit_standardize == 'cross_sectional'``); ``±inf`` / missing → the (0) mean.
    Training rows whose forward-return label leaves the IS window are dropped, so
    the fit never consumes validation/test outcomes through boundary labels.
    Returns ``None`` if there are too few finite IS training rows to fit.
    """
    from quant_fund_agent.backtesting.strategy_backtester import normalise_factor_signals
    from quant_fund_agent.modeling.catalog import build_estimator

    if not signals:
        return None
    index, cols = close.index, close.columns
    n_rows, n_cols = len(index), len(cols)
    is_idx = index[is_mask]

    feats = []
    for sig in signals:
        s = sig.reindex(index=index, columns=cols).replace([np.inf, -np.inf], np.nan)
        if getattr(cfg, "fit_standardize", "per_underlying") == "cross_sectional":
            z = normalise_factor_signals({"_": s})["_"]
        else:
            z = per_underlying_zscore(s, is_idx)
        feats.append(z.to_numpy(dtype=float).ravel())
    X = np.nan_to_num(np.column_stack(feats), nan=0.0, posinf=0.0, neginf=0.0)
    y = forward_returns(close, horizon=cfg.target_horizon).to_numpy(dtype=float).ravel()

    fit_mask = np.asarray(is_mask) & _label_available_mask(is_mask, cfg.target_horizon)
    is_rows = np.repeat(fit_mask, n_cols)
    train = np.flatnonzero(is_rows & np.isfinite(y))
    if len(train) < 30:
        return None
    fast = cfg.fast_model_params(model) if hasattr(cfg, "fast_model_params") else None
    est = build_estimator(model, fast)
    est.fit(X[train], y[train])
    pred = np.asarray(est.predict(X), dtype=float).reshape(n_rows, n_cols)
    return pd.DataFrame(pred, index=index, columns=cols)


def _label_available_mask(available_mask: np.ndarray, horizon: int) -> np.ndarray:
    """Rows whose forward-return label stays inside ``available_mask``.

    With a three-way split, VALIDATION's final ``horizon`` rows would otherwise
    use prices from TEST to form labels.  They are unavailable in a real
    "train today, deploy tomorrow" workflow, so they must not participate in
    any research-time score.
    """
    available = np.asarray(available_mask, dtype=bool)
    n = len(available)
    h = max(0, int(horizon))
    ok = available.copy()
    if h > 0:
        future_ok = np.zeros(n, dtype=bool)
        if h < n:
            future_ok[: n - h] = available[h:]
        ok &= future_ok
    return ok


def _flat_z(
    sig: pd.DataFrame,
    close: pd.DataFrame,
    *,
    stat_mask: np.ndarray | None = None,
    row_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Per-underlying z-scored signal, reindexed to the panel grid and flattened."""
    s = sig.reindex(index=close.index, columns=close.columns)
    stat_idx = close.index[stat_mask] if stat_mask is not None else None
    z = per_underlying_zscore(s, stat_idx).to_numpy(dtype=float)
    if row_mask is not None:
        z = z[np.asarray(row_mask)]
    return z.ravel()


# ── the axes ──────────────────────────────────────────────────────────────────

def _marginal_value(
    candidate: pd.DataFrame, book: Sequence[pd.DataFrame], panel, cfg,
    is_mask, val_mask, model: str, available_mask: np.ndarray,
) -> dict[str, Any]:
    """LOCO: ΔOOS-IC of the combined model from adding the candidate to the book.

    The fitted ``with``/``base`` predictions are returned alongside the scalar so
    the regime axis can re-score them on the crash bars without a second fit.
    """
    close = panel["close"]
    h = cfg.target_horizon

    with_pred = _combined_prediction([*book, candidate], close, is_mask, cfg, model)
    with_ic = (
        _pooled_ic(with_pred, close, h, val_mask, is_mask, available_mask)[0]
        if with_pred is not None else None
    )

    if book:
        base_pred = _combined_prediction(list(book), close, is_mask, cfg, model)
        base_ic = (
            _pooled_ic(base_pred, close, h, val_mask, is_mask, available_mask)[0]
            if base_pred is not None else None
        )
    else:
        base_pred = None
        base_ic = 0.0  # empty book → the candidate is the whole edge

    if with_ic is None:
        return {"marginal_value": None, "with_ic": None, "base_ic": base_ic,
                "with_pred": with_pred, "base_pred": base_pred}
    # The combined model predicts the forward return, so a useful signal has a
    # POSITIVE IC; ΔIC then measures the edge the candidate adds on top of the book.
    marginal = with_ic - (base_ic if base_ic is not None else 0.0)
    return {"marginal_value": marginal, "with_ic": with_ic, "base_ic": base_ic,
            "with_pred": with_pred, "base_pred": base_pred}


def _distribution_stats(values: Sequence[float]) -> tuple[float | None, float | None]:
    """Mean/std pair for a fold-score distribution, requiring at least two folds."""
    if len(values) < 2:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def _cpcv_fit_cfg(cfg: Any, params: EvalParams) -> Any:
    """Config used for repeated CPCV refits.

    CPCV now refits one model per fold, so by default it flips the existing
    comparison ``fast`` preset on for tree/boosting models.  The one-shot
    marginal-value axis still uses the caller's original config unchanged.
    """
    if not params.cpcv_fast:
        return cfg
    try:
        return replace(cfg, fast=True)
    except Exception:  # non-dataclass config-like object; fall back quietly
        return cfg


def _standalone_cpcv_ics(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    h: int,
    folds,
    available_mask: np.ndarray,
    sign: float,
) -> list[float]:
    """Raw-signal CPCV IC stability diagnostic (no model fit)."""
    ics: list[float] = []
    for fold in folds:
        ic, _ = _pooled_ic(signal, close, h, fold.test, fold.train, available_mask)
        if ic is not None:
            ics.append(ic * sign)
    return ics


def _refit_cpcv_scores(
    signals_with: Sequence[pd.DataFrame],
    signals_base: Sequence[pd.DataFrame],
    close: pd.DataFrame,
    cfg: Any,
    params: EvalParams,
    folds,
    h: int,
    available_mask: np.ndarray,
    sign: float,
) -> dict[str, Any]:
    """Fold-refit combined-model CPCV scores.

    For SINGLE mode ``signals_with`` is ``book + candidate`` and
    ``signals_base`` is ``book``; each fold score is LOCO ``with_ic - base_ic``.
    For SET mode ``signals_base`` is empty; each fold score is the set's own
    combined-model fold-test IC.  Base predictions are cached per fold inside the
    candidate evaluation so the same base model is not fitted twice.
    """
    model = params.cpcv_model or params.marginal_model
    fit_cfg = _cpcv_fit_cfg(cfg, params)
    scores: list[float] = []
    with_ics: list[float] = []
    base_ics: list[float] = []
    base_pred_cache: dict[tuple[int, ...], pd.DataFrame | None] = {}

    for fold in folds:
        with_pred = _combined_prediction(signals_with, close, fold.train, fit_cfg, model)
        if with_pred is None:
            continue
        with_ic = _pooled_ic(with_pred, close, h, fold.test, fold.train,
                             available_mask)[0]
        if with_ic is None:
            continue

        if signals_base:
            key = tuple(fold.test_groups)
            if key not in base_pred_cache:
                base_pred_cache[key] = _combined_prediction(
                    signals_base, close, fold.train, fit_cfg, model)
            base_pred = base_pred_cache[key]
            base_ic = (_pooled_ic(base_pred, close, h, fold.test, fold.train,
                                  available_mask)[0]
                       if base_pred is not None else None)
            score = with_ic - (base_ic if base_ic is not None else 0.0)
            if base_ic is not None:
                base_ics.append(base_ic * sign)
        else:
            base_ic = None
            score = with_ic

        with_ics.append(with_ic * sign)
        scores.append(score * sign)

    mean, std = _distribution_stats(scores)
    with_mean, with_std = _distribution_stats(with_ics)
    base_mean, base_std = _distribution_stats(base_ics)
    kind = "refit_marginal_delta" if signals_base else "refit_combined"
    return {
        "scores": scores,
        "mean": mean,
        "std": std,
        "n_folds": len(scores),
        "with_ic_mean": with_mean,
        "with_ic_std": with_std,
        "base_ic_mean": base_mean,
        "base_ic_std": base_std,
        "score_kind": kind,
        "model": model,
        "fast": bool(params.cpcv_fast),
    }


def _stress_mask(close: pd.DataFrame, params: EvalParams,
                 scope_mask: np.ndarray) -> np.ndarray:
    """Boolean mask of "stress"/"crash" bars, computed within ``scope_mask`` only.

    ``drawdown`` (default): the worst ``regime_quantile`` fraction of bars by
    cross-sectional mean return (the crash bars).  ``volatility``: the top
    ``regime_quantile`` fraction by rolling market-return volatility.  The
    labelling quantile is taken over the development window (``scope_mask``,
    i.e. IS∪VAL) only and never reads a held-out TEST price, so the regime axis
    stays leak-free exactly like every other score.
    """
    scope = np.asarray(scope_mask, dtype=bool)
    mkt = close.pct_change().mean(axis=1)  # cross-sectional mean simple return / bar
    high = getattr(params, "regime_kind", "drawdown") == "volatility"
    if high:
        window = max(2, int(params.regime_vol_window))
        metric = mkt.rolling(window, min_periods=max(2, window // 2)).std() \
            .to_numpy(dtype=float)
    else:
        metric = mkt.to_numpy(dtype=float)

    finite = np.isfinite(metric) & scope
    vals = metric[finite]
    q = min(max(float(params.regime_quantile), 1e-6), 0.9)
    if vals.size < max(5, int(round(1.0 / q))):
        return np.zeros(len(scope), dtype=bool)  # too few bars to label a regime
    if high:
        thr = float(np.quantile(vals, 1.0 - q))
        stress = metric >= thr
    else:
        thr = float(np.quantile(vals, q))
        stress = metric <= thr
    return np.asarray(stress) & finite


def _regime_complementarity(
    marg: dict[str, Any], close: pd.DataFrame, h: int,
    val_mask: np.ndarray, stress_mask: np.ndarray, is_mask: np.ndarray,
    available_mask: np.ndarray, has_book: bool, params: EvalParams,
) -> dict[str, Any]:
    """Marginal ΔIC of the combined model on the VAL∩stress bars (crash edge).

    Reuses the ``with``/``base`` predictions already fitted for the marginal-value
    axis, re-scored on the stress bars only.  Positive & large ⇒ the candidate
    improves the book's predictions exactly where the book is weakest.  Empty book
    ⇒ the candidate's own crash-period IC.  ``None`` when there are too few stress
    observations to trust the IC.
    """
    with_pred = marg.get("with_pred")
    base_pred = marg.get("base_pred")
    empty = {"regime_independence": None, "stress_ic_with": None,
             "stress_ic_base": None, "n_stress_obs": 0}
    if with_pred is None:
        return empty
    val_stress = np.asarray(val_mask, dtype=bool) & np.asarray(stress_mask, dtype=bool)
    if not val_stress.any():
        return empty
    sw, n_with = _pooled_ic(with_pred, close, h, val_stress, is_mask, available_mask)
    if sw is None or n_with < params.regime_min_obs:
        return {"regime_independence": None, "stress_ic_with": sw,
                "stress_ic_base": None, "n_stress_obs": int(n_with or 0)}
    if has_book and base_pred is not None:
        sb = _pooled_ic(base_pred, close, h, val_stress, is_mask, available_mask)[0]
    else:
        sb = 0.0  # empty book → the candidate is the whole crash edge
    regime = sw - (sb if sb is not None else 0.0)
    return {"regime_independence": regime, "stress_ic_with": sw,
            "stress_ic_base": sb, "n_stress_obs": int(n_with)}


def _residual_ic(
    candidate: pd.DataFrame,
    book: Sequence[pd.DataFrame],
    panel,
    cfg,
    fit_mask: np.ndarray,
    score_mask: np.ndarray,
    available_mask: np.ndarray,
) -> float | None:
    """Orthogonalised (residual) IC — the candidate's edge beyond the book's span.

    Regress the candidate on the book using ``fit_mask`` only, then
    Spearman-correlate the residual with forward returns on ``score_mask``.
    ``available_mask`` prevents a score near the development boundary from using
    held-out TEST prices as forward-return labels.
    """
    close = panel["close"]
    h = cfg.target_horizon
    if not book:
        return _pooled_ic(
            candidate, close, h, score_mask, fit_mask, available_mask
        )[0]

    y_fwd = forward_returns(close, horizon=h).to_numpy(dtype=float).ravel()
    cand = _flat_z(candidate, close, stat_mask=fit_mask)

    B = np.column_stack([_flat_z(s, close, stat_mask=fit_mask) for s in book])
    n_cols = close.shape[1]
    fit_rows = np.repeat(np.asarray(fit_mask), n_cols)
    finite_fit = fit_rows & np.isfinite(cand) & np.isfinite(B).all(axis=1)
    if finite_fit.sum() < 30:
        return None
    B_fit = np.column_stack([np.ones(finite_fit.sum()), B[finite_fit]])
    beta, *_ = np.linalg.lstsq(B_fit, cand[finite_fit], rcond=None)

    resid = np.full_like(cand, np.nan)
    finite_all = np.isfinite(cand) & np.isfinite(B).all(axis=1)
    B_all = np.column_stack([np.ones(finite_all.sum()), B[finite_all]])
    resid[finite_all] = cand[finite_all] - B_all @ beta

    label_rows = _label_available_mask(available_mask, h)
    score_rows = np.repeat(np.asarray(score_mask) & label_rows, n_cols)
    finite_score = score_rows & np.isfinite(resid) & np.isfinite(y_fwd)
    return _spearman(resid[finite_score], y_fwd[finite_score])[0]


def _independence(candidate: pd.DataFrame, book: Sequence[pd.DataFrame], panel, cfg,
                  corr_penalty: float, mask: np.ndarray) -> dict[str, Any]:
    """Δ participation ratio (book vs book+candidate) minus a soft max-|corr| penalty."""
    close = panel["close"]
    cand = _flat_z(candidate, close, stat_mask=mask, row_mask=mask)

    if not book:
        return {"independence": 1.0, "delta_participation": 1.0, "max_abs_corr": 0.0,
                "pr_before": 0.0, "pr_after": 1.0}

    cols = {
        f"b{i}": _flat_z(s, close, stat_mask=mask, row_mask=mask)
        for i, s in enumerate(book)
    }
    before = pd.DataFrame(cols).corr().to_numpy(dtype=float)
    pr_before = participation_ratio(before) or float(len(book))

    cols_after = dict(cols)
    cols_after["cand"] = cand
    after_df = pd.DataFrame(cols_after)
    after = after_df.corr().to_numpy(dtype=float)
    pr_after = participation_ratio(after) or float(len(book) + 1)

    corr_row = after_df.corr()["cand"].drop("cand").abs()
    max_abs_corr = float(corr_row.max()) if len(corr_row) else 0.0
    delta_pr = pr_after - pr_before
    return {"independence": delta_pr - corr_penalty * max_abs_corr,
            "delta_participation": delta_pr, "max_abs_corr": max_abs_corr,
            "pr_before": pr_before, "pr_after": pr_after}


def _robustness(
    candidate: pd.DataFrame,
    panel,
    cfg,
    split: ThreeWaySplit,
    params: EvalParams,
    overall_ic: float | None,
    expected_sign: int | None,
    jitter_signals: Sequence[pd.DataFrame] | None = None,
    *,
    refit_signals: Sequence[pd.DataFrame] | None = None,
    refit_base_signals: Sequence[pd.DataFrame] | None = None,
    standalone_overall_ic: float | None = None,
) -> dict[str, Any]:
    """CPCV robustness axis + raw-signal stability diagnostic.

    The scored axis refits a combined model on every CPCV fold train mask and
    scores the fold test mask.  The previous raw-signal CPCV distribution is
    retained as a diagnostic because it is still useful temporal-stability
    evidence, but it no longer drives Pareto robustness in the fitted-book path.
    """
    close = panel["close"]
    h = cfg.target_horizon
    sign = 1.0 if (overall_ic is None or overall_ic >= 0) else -1.0
    standalone_ref = standalone_overall_ic if standalone_overall_ic is not None else overall_ic
    standalone_sign = 1.0 if (standalone_ref is None or standalone_ref >= 0) else -1.0

    folds = cpcv_folds(close.index, n_groups=params.cpcv_groups, k_test=params.cpcv_k,
                       horizon=h, embargo=params.embargo, mask=split.is_val_mask)
    standalone_ics = _standalone_cpcv_ics(
        candidate, close, h, folds, split.is_val_mask, standalone_sign)
    standalone_mean, standalone_std = _distribution_stats(standalone_ics)

    if refit_signals:
        refit = _refit_cpcv_scores(
            refit_signals, refit_base_signals or [], close, cfg, params, folds, h,
            split.is_val_mask, sign)
    else:
        # Defensive fallback for legacy/direct callers: still return a usable
        # robustness axis even if no fit context was supplied.
        refit = {
            "scores": standalone_ics,
            "mean": standalone_mean,
            "std": standalone_std,
            "n_folds": len(standalone_ics),
            "with_ic_mean": standalone_mean,
            "with_ic_std": standalone_std,
            "base_ic_mean": None,
            "base_ic_std": None,
            "score_kind": "standalone_ic",
            "model": None,
            "fast": False,
        }

    cpcv_mean = refit["mean"]
    cpcv_std = refit["std"]
    robustness = (
        None if cpcv_mean is None or cpcv_std is None
        else cpcv_mean - params.lambda_std * cpcv_std
    )

    # Parameter-sensitivity plateau test: score each window-jittered variant on
    # VAL exactly like the candidate; the penalty is the (sign-aligned) IC drop
    # from the candidate to the jitter mean, floored at 0 — a knife-edge factor
    # loses its edge under a ±10% window change, a plateau factor keeps it.
    plateau_penalty = None
    jitter_ics: list[float] = []
    if jitter_signals and standalone_ref is not None:
        for jsig in jitter_signals:
            jic = _pooled_ic(
                jsig, close, h, split.val_mask, split.is_mask, split.is_val_mask
            )[0]
            if jic is not None:
                jitter_ics.append(jic * standalone_sign)
        if jitter_ics:
            plateau_penalty = max(
                0.0, float(standalone_ref * standalone_sign - np.mean(jitter_ics)))
            if robustness is not None:
                robustness -= params.plateau_weight * plateau_penalty

    # Perturbation-fidelity probe (AlphaEval), folded into robustness like the plateau
    # penalty: shock the signal with Gaussian noise (in z-units) and measure the
    # sign-aligned VAL-IC retained.  A knife-edge / noise-sensitive factor loses its
    # edge under the shock and is docked; a robust one keeps it.  Off when weight == 0.
    perturbation_penalty = None
    if params.perturbation_weight > 0 and standalone_ref is not None and robustness is not None:
        rng = np.random.default_rng(0)  # fixed seed → deterministic, un-gameable
        s = candidate.reindex(index=close.index, columns=close.columns)
        # noise scale from the DEV window only (leak-free — never reads TEST rows)
        dev = s.to_numpy(dtype=float)[np.asarray(split.is_val_mask, dtype=bool)]
        col_std = np.nanstd(dev, axis=0) if dev.size else np.zeros(s.shape[1])
        col_std = np.where(np.isfinite(col_std) & (col_std > 0), col_std, np.nan)
        pert_ics: list[float] = []
        for _ in range(max(1, int(params.perturbation_draws))):
            noise = rng.standard_normal(s.shape) * params.perturbation_sigma * col_std
            pic = _pooled_ic(s + noise, close, h, split.val_mask, split.is_mask,
                             split.is_val_mask)[0]
            if pic is not None:
                pert_ics.append(pic * standalone_sign)
        if pert_ics:
            perturbation_penalty = max(
                0.0, float(standalone_ref * standalone_sign - np.mean(pert_ics)))
            robustness -= params.perturbation_weight * perturbation_penalty

    sign_consistency = None
    if expected_sign is not None and standalone_ref is not None:
        realized = 1 if standalone_ref >= 0 else -1
        sign_consistency = bool(realized == int(np.sign(expected_sign) or 1))
        if robustness is not None:
            robustness += params.sign_bonus if sign_consistency else -params.sign_bonus

    return {"robustness": robustness, "cpcv_ic_mean": cpcv_mean, "cpcv_ic_std": cpcv_std,
            "cpcv_n_folds": refit["n_folds"], "sign_consistency": sign_consistency,
            "plateau_penalty": plateau_penalty, "jitter_ics": jitter_ics,
            "perturbation_penalty": perturbation_penalty,
            "cpcv_score_kind": refit["score_kind"], "cpcv_model": refit["model"],
            "cpcv_fast": refit["fast"],
            "cpcv_with_ic_mean": refit["with_ic_mean"],
            "cpcv_with_ic_std": refit["with_ic_std"],
            "cpcv_base_ic_mean": refit["base_ic_mean"],
            "cpcv_base_ic_std": refit["base_ic_std"],
            "standalone_cpcv_ic_mean": standalone_mean,
            "standalone_cpcv_ic_std": standalone_std,
            "standalone_cpcv_n_folds": len(standalone_ics)}


def _behavior_descriptors(sig: pd.DataFrame, close: pd.DataFrame, h: int,
                          split: ThreeWaySplit, params: EvalParams) -> dict[str, Any]:
    """QD behavior descriptors (WS2) — the candidate's *behavioral* coordinates.

    These are NOT fitness axes (they never enter the objective vector or a gate); they
    place the candidate in the MAP-Elites grid so the search fills a diverse library.
    All computed on the development window (IS∪VAL) only, so they are leak-free:

    * ``trend_reversal`` — Spearman(signal, *trailing* h-bar return).  Negative ⇒
      mean-reversion / fade, positive ⇒ momentum / continuation.  (Trailing, not
      forward — a behavioral property, not predictive power.)
    * ``signal_speed`` — ``1 − lag-1 autocorr`` of the per-underlying z-signal.  Low ⇒
      slow / state-like, high ⇒ fast / reactive.
    * ``stress_activation`` — ``mean(|z| on stress bars) / mean(|z| on normal bars)``:
      how much more the factor "fires" in crashes.  Metadata at ``grid_dims=2``.
    """
    dev = np.asarray(split.is_val_mask, dtype=bool)
    dev_idx = close.index[dev]
    z = per_underlying_zscore(sig.reindex(index=close.index, columns=close.columns), dev_idx)
    zt = z.to_numpy(dtype=float)

    # trend_reversal: signal vs trailing h-bar return (past return, leak-free)
    trailing = close.pct_change(max(1, int(h))).to_numpy(dtype=float)
    tr = _spearman(zt[dev].ravel(), trailing[dev].ravel())[0]

    # signal_speed: 1 − mean per-underlying lag-1 autocorr over the dev window
    acs: list[float] = []
    for j in range(zt.shape[1]):
        col = zt[dev, j]
        col = col[np.isfinite(col)]
        if len(col) > 10 and np.std(col) > 0:
            ac = float(np.corrcoef(col[:-1], col[1:])[0, 1])
            if np.isfinite(ac):
                acs.append(ac)
    signal_speed = (1.0 - float(np.mean(acs))) if acs else None

    # stress_activation: |z| firing in stress vs normal dev bars
    stress = _stress_mask(close, params, split.is_val_mask)
    zabs = np.abs(zt)
    s_rows = dev & np.asarray(stress, dtype=bool)
    n_rows = dev & ~np.asarray(stress, dtype=bool)
    sa = None
    if s_rows.any() and n_rows.any():
        s_mean = np.nanmean(zabs[s_rows])
        n_mean = np.nanmean(zabs[n_rows])
        if np.isfinite(s_mean) and np.isfinite(n_mean) and n_mean > 0:
            sa = float(s_mean / n_mean)

    return {"trend_reversal": tr, "signal_speed": signal_speed,
            "stress_activation": sa}


def _coverage(sig: pd.DataFrame, close: pd.DataFrame, mask: np.ndarray) -> float:
    """Fraction of (date,ticker) cells in ``mask`` where the signal is finite."""
    s = sig.reindex(index=close.index, columns=close.columns).to_numpy(dtype=float)[mask]
    if s.size == 0:
        return 0.0
    return float(np.isfinite(s).mean())


def _turnover_netcost(sig: pd.DataFrame, close: pd.DataFrame, h: int,
                      mask: np.ndarray, params: EvalParams) -> dict[str, Any]:
    """Factor-level turnover + net-of-cost P&L via an EXPLICIT position construction.

    Positions come from the SHARED per-underlying primitives
    (``backtesting.positions``: causal per-underlying z-score → directional
    long/flat/short band), so "turnover" means byte-identically the same thing here,
    in the comparison track and in the deployed fund.  Deliberately *not*
    ``vector_backtest`` (whose IS/OOS split + internal position choices are heavier
    and implicit).  Returns per-bar turnover (mean ``|Δposition|``) and the gross /
    net-of-cost mean P&L over ``mask`` (net = ``position·fwd − cost_rate·|Δposition|``).
    """
    from quant_fund_agent.backtesting.positions import (
        directional_positions,
        zscore_over_time,
    )

    s = sig.reindex(index=close.index, columns=close.columns)
    z = zscore_over_time(s, params.turnover_zscore_basis, 500)
    pos = directional_positions(z, mode=params.turnover_position_mode,
                                threshold=params.turnover_position_threshold)
    dpos = pos.diff().abs()
    fwd = forward_returns(close, horizon=h)
    m = np.asarray(mask, dtype=bool)
    pos_a = pos.to_numpy(dtype=float)
    dp_a = np.where(np.isfinite(dpos.to_numpy(dtype=float)), dpos.to_numpy(dtype=float), 0.0)
    fwd_a = fwd.to_numpy(dtype=float)

    # Turnover is a causal, position-only property → over the whole dev window ``m``
    # (the expanding z-score never reads future rows, so this is leak-free).
    tpos = pos_a[m]
    tdp = dp_a[m]
    turnover = (float(np.mean(tdp[np.isfinite(tpos)]))
                if np.isfinite(tpos).any() else None)

    # P&L uses forward-return labels → drop rows whose ``t+h`` label leaves the dev
    # window (else the last ``h`` VAL rows would consume held-out TEST prices), exactly
    # like every other scored axis.
    label_ok = m & _label_available_mask(m, h)
    ppos, pdp, pfwd = pos_a[label_ok], dp_a[label_ok], fwd_a[label_ok]
    finite = np.isfinite(ppos) & np.isfinite(pfwd)
    if not finite.any():
        return {"turnover": turnover, "gross_ret": None, "net_ret": None,
                "net_gross_ratio": None}
    gross = ppos[finite] * pfwd[finite]
    net = gross - params.cost_rate * pdp[finite]
    gross_m = float(np.mean(gross))
    net_m = float(np.mean(net))
    ratio = (net_m / gross_m) if gross_m not in (0.0,) else None
    return {"turnover": turnover, "gross_ret": gross_m, "net_ret": net_m,
            "net_gross_ratio": ratio}


def _zoo_dedup(candidate: pd.DataFrame, candidate_code: str | None,
               reference_signals: Sequence[pd.DataFrame] | None,
               reference_ids: Sequence[str] | None,
               reference_codes: Sequence[str] | None,
               close: pd.DataFrame, mask: np.ndarray) -> dict[str, Any]:
    """Novelty vs a REFERENCE factor zoo (e.g. the 86 base factors) — DIAG only.

    Reports the candidate's max ``|signal correlation|`` to any reference factor (the
    "did we just rediscover a known alpha?" flag, à la AlphaAgent's AST-originality)
    and the minimum normalised code distance (``1 − difflib ratio``; 0 = identical
    source).  Deliberately **not a gate** yet — a diagnostic the author can point at a
    chosen reference set (the ~86 base factors) and later promote to a penalty/gate.
    """
    out: dict[str, Any] = {"zoo_max_abs_corr": None, "zoo_nearest": None,
                           "zoo_min_code_distance": None}
    if reference_signals:
        cand = _flat_z(candidate, close, stat_mask=mask, row_mask=mask)
        best_corr: float | None = None
        best_idx: int | None = None
        for i, ref in enumerate(reference_signals):
            r = _flat_z(ref, close, stat_mask=mask, row_mask=mask)
            ok = np.isfinite(cand) & np.isfinite(r)
            if ok.sum() < 10 or np.std(cand[ok]) == 0 or np.std(r[ok]) == 0:
                continue
            c = abs(float(np.corrcoef(cand[ok], r[ok])[0, 1]))
            if best_corr is None or c > best_corr:
                best_corr, best_idx = c, i
        out["zoo_max_abs_corr"] = best_corr
        if best_idx is not None:
            out["zoo_nearest"] = (reference_ids[best_idx]
                                  if reference_ids and best_idx < len(reference_ids)
                                  else int(best_idx))
    if candidate_code and reference_codes:
        import difflib
        cc = "".join(candidate_code.split())
        dists = [1.0 - difflib.SequenceMatcher(None, cc, "".join((rc or "").split())).ratio()
                 for rc in reference_codes if rc and rc.strip()]
        if dists:
            out["zoo_min_code_distance"] = float(min(dists))
    return out


# ── the orchestrator ──────────────────────────────────────────────────────────

def evaluate_candidate(
    candidate_signal: pd.DataFrame,
    book_signals: Sequence[pd.DataFrame],
    panel: dict[str, Any],
    cfg: Any,
    split: ThreeWaySplit | None = None,
    *,
    params: EvalParams | None = None,
    candidate_code: str | None = None,
    expected_sign: int | None = None,
    candidate_id: str = "candidate",
    jitter_signals: Sequence[pd.DataFrame] | None = None,
    reference_signals: Sequence[pd.DataFrame] | None = None,
    reference_ids: Sequence[str] | None = None,
    reference_codes: Sequence[str] | None = None,
) -> FitnessResult:
    """Score one candidate signal against the current book → a :class:`FitnessResult`.

    ``split`` is the IS/VAL/TEST seam: the combined model fits on **IS**, the fitness
    (marginal value, gates) is measured on **VAL** (deliberately burned by the
    search), CPCV runs over **IS∪VAL**, and TEST is never touched here.  ``None``
    derives a default 60/20/20 fraction split.  ``params`` carries the deterministic
    scoring knobs (including ``n_trials`` for deflation); ``candidate_code`` feeds the
    parsimony axis; ``expected_sign`` (from the Hypothesis agent, later phases) feeds
    the sign-consistency check; ``jitter_signals`` (signals of window-jittered
    variants of the candidate) feeds the plateau penalty inside robustness.
    """
    params = params or EvalParams()
    close = panel["close"]
    if split is None:
        split = three_way_split(close.index)
    is_mask, val_mask = split.is_mask, split.val_mask
    h = cfg.target_horizon
    book = list(book_signals)

    # ── Family 1 — standalone power (DIAG) ──
    ic_decay = {}
    for hz in params.ic_decay_horizons:
        ic_decay[str(hz)] = _pooled_ic(
            candidate_signal, close, hz, val_mask, is_mask, split.is_val_mask
        )[0]
    standalone_ic = ic_decay.get(str(h))
    if standalone_ic is None:
        standalone_ic = _pooled_ic(
            candidate_signal, close, h, val_mask, is_mask, split.is_val_mask
        )[0]

    is_ic, is_n = _pooled_ic(
        candidate_signal, close, h, is_mask, is_mask, is_mask
    )
    val_ic, val_n = _pooled_ic(
        candidate_signal, close, h, val_mask, is_mask, split.is_val_mask
    )

    # ── Family 2 — marginal / incremental value (CORE, primary) ──
    marg = _marginal_value(candidate_signal, book, panel, cfg, is_mask, val_mask,
                           params.marginal_model, split.is_val_mask)
    residual_ic = _residual_ic(
        candidate_signal, book, panel, cfg, is_mask, val_mask, split.is_val_mask
    )

    # ── Family 3 — independence (CORE) ──
    # Axis = residual (orthogonalised) IC by default — the candidate's predictive
    # edge in the direction the book does not span (novel content that *predicts*).
    # The legacy Δ-participation-ratio − max-|corr| penalty stays computed as a
    # diagnostic and is selectable via `independence_metric`.
    indep = _independence(
        candidate_signal, book, panel, cfg, params.corr_penalty, split.is_val_mask
    )
    independence_axis = (
        residual_ic if params.independence_metric == "residual_ic"
        else indep["independence"]
    )

    # ── Family 4 — robustness (CORE) ──
    robust = _robustness(
        candidate_signal, panel, cfg, split, params, marg["marginal_value"],
        expected_sign, jitter_signals,
        refit_signals=[*book, candidate_signal],
        refit_base_signals=book,
        standalone_overall_ic=val_ic,
    )

    # ── regime axis (CORE) — marginal ΔIC on the crash bars (book-conditioned) ──
    stress_mask = _stress_mask(close, params, split.is_val_mask)
    regime = _regime_complementarity(
        marg, close, h, val_mask, stress_mask, is_mask, split.is_val_mask,
        bool(book), params)

    # ── parsimony axis ──
    parsimony = -float(complexity(candidate_code)) if candidate_code is not None else None

    # ── factor-zoo dedup (WS4) — novelty vs a reference book, DIAG only ──
    zoo = _zoo_dedup(candidate_signal, candidate_code, reference_signals,
                     reference_ids, reference_codes, close, split.is_val_mask)

    # ── hard gates ──
    coverage = _coverage(candidate_signal, close, split.is_val_mask)
    reasons: dict[str, str] = {}

    coverage_ok = coverage >= params.gate_coverage
    if not coverage_ok:
        reasons["coverage"] = f"{coverage:.3f} < τ={params.gate_coverage}"

    # OOS/IS degradation: aligned by the IS sign, require same sign + ratio ≥ τ.
    degradation_ok: bool | None
    deg_ratio = None
    if is_ic is None or val_ic is None or abs(is_ic) < params.min_is_ic:
        degradation_ok = None  # not evaluable on a near-zero IS edge
    else:
        deg_ratio = (val_ic * np.sign(is_ic)) / abs(is_ic)
        degradation_ok = bool(deg_ratio >= params.gate_degradation)
        if not degradation_ok:
            reasons["degradation"] = f"OOS/IS={deg_ratio:.3f} < τ={params.gate_degradation}"

    # N_trials-aware deflation is computed as a DIAGNOSTIC only (teacher channel).
    # It is deliberately NOT a per-candidate *search* gate any more: deflation is a
    # *selection*-time control applied ONCE, on the combined-book statistic, by
    # ``research_eval.publish`` (WS1).  Keeping it out of the search gates means a
    # candidate that would fail deflation in isolation still stays ``search_selectable``
    # (a valid parent / archive member) — the "discovery" half of the two modes —
    # while multiple-testing honesty is enforced at publish (the "validation" half).
    # ``deflation_ok`` therefore stays ``None`` (does not count against the candidate).
    defl = deflation.deflated_ic(abs(val_ic) if val_ic is not None else None,
                                 int(val_n), params.n_trials)

    # ── economic-realism gate (P5, WS3): turnover / net-of-cost, folded in ──
    tc = _turnover_netcost(candidate_signal, close, h, split.is_val_mask, params)
    cost_ok: bool | None = None
    if params.gate_turnover is not None and tc["turnover"] is not None:
        cost_ok = bool(tc["turnover"] <= params.gate_turnover)
        if not cost_ok:
            reasons["cost"] = f"turnover={tc['turnover']:.3f} > τ={params.gate_turnover}"

    gates = GateResults(coverage_ok=coverage_ok, degradation_ok=degradation_ok,
                        deflation_ok=None, cost_ok=cost_ok, reasons=reasons)

    objective = ObjectiveVector(
        marginal_value=marg["marginal_value"],
        independence=independence_axis,
        robustness=robust["robustness"],
        parsimony=parsimony,
        regime_independence=regime["regime_independence"],
    )

    diagnostics = {
        "standalone_ic": standalone_ic,
        "ic_decay": ic_decay,
        "is_ic": is_ic, "val_ic": val_ic, "val_n_obs": int(val_n),
        "residual_ic": residual_ic,
        "independence_metric": params.independence_metric,
        "with_ic": marg["with_ic"], "base_ic": marg["base_ic"],
        "max_abs_corr": indep["max_abs_corr"],
        "delta_participation": indep["delta_participation"],
        "pr_before": indep["pr_before"], "pr_after": indep["pr_after"],
        "cpcv_ic_mean": robust["cpcv_ic_mean"], "cpcv_ic_std": robust["cpcv_ic_std"],
        "cpcv_n_folds": robust["cpcv_n_folds"],
        "cpcv_score_kind": robust["cpcv_score_kind"],
        "cpcv_model": robust["cpcv_model"],
        "cpcv_fast": robust["cpcv_fast"],
        "cpcv_with_ic_mean": robust["cpcv_with_ic_mean"],
        "cpcv_with_ic_std": robust["cpcv_with_ic_std"],
        "cpcv_base_ic_mean": robust["cpcv_base_ic_mean"],
        "cpcv_base_ic_std": robust["cpcv_base_ic_std"],
        "standalone_cpcv_ic_mean": robust["standalone_cpcv_ic_mean"],
        "standalone_cpcv_ic_std": robust["standalone_cpcv_ic_std"],
        "standalone_cpcv_n_folds": robust["standalone_cpcv_n_folds"],
        "sign_consistency": robust["sign_consistency"],
        "plateau_penalty": robust["plateau_penalty"],
        "jitter_ics": robust["jitter_ics"],
        "perturbation_penalty": robust.get("perturbation_penalty"),
        "turnover": tc["turnover"],
        "gross_ret": tc["gross_ret"],
        "net_ret": tc["net_ret"],
        "net_gross_ratio": tc["net_gross_ratio"],
        "zoo_max_abs_corr": zoo["zoo_max_abs_corr"],
        "zoo_nearest": zoo["zoo_nearest"],
        "zoo_min_code_distance": zoo["zoo_min_code_distance"],
        "regime_independence": regime["regime_independence"],
        "regime_kind": params.regime_kind,
        "stress_ic_with": regime["stress_ic_with"],
        "stress_ic_base": regime["stress_ic_base"],
        "n_stress_obs": regime["n_stress_obs"],
        "coverage": coverage,
        "degradation_ratio": (float(deg_ratio) if deg_ratio is not None else None),
        "deflation": defl,
        "n_trials": params.n_trials,
        "complexity": (complexity(candidate_code) if candidate_code is not None else None),
    }

    behavior = _behavior_descriptors(candidate_signal, close, h, split, params)

    return FitnessResult(candidate_id=candidate_id, objective=objective, gates=gates,
                         diagnostics=diagnostics, behavior=behavior,
                         raw={"split_sizes": split.sizes})


# ── SET mode (P5): evaluate a whole "alpha program" jointly ───────────────────

def evaluate_set(
    member_signals: dict[str, pd.DataFrame],
    panel: dict[str, Any],
    cfg: Any,
    split: ThreeWaySplit | None = None,
    *,
    params: EvalParams | None = None,
    member_codes: dict[str, str] | None = None,
    expected_sign: int | None = None,
    candidate_id: str = "set",
) -> FitnessResult:
    """Score a factor SET as one genome (the AlphaEvolve-style unit).

    The Pareto axes change meaning per the design table:

    1. ``marginal_value`` — the set's **own** combined-model OOS IC (fit on IS,
       scored on VAL); no LOCO against an external archive — the set is
       self-contained.
    2. ``independence`` — the set's *internal* participation ratio (effective
       number of independent members), normalised by size to ``[0, 1]`` so sets
       of different sizes compare fairly.
    3. ``robustness`` — the same ``mean − λ·std`` CPCV formula, computed on the
       **combined signal**.
    4. ``parsimony`` — ``−(total member complexity / set size)``.

    Gates (coverage / degradation / deflation) also apply to the combined
    signal.  Reuses every helper the SINGLE path uses, so the two modes cannot
    drift.
    """
    params = params or EvalParams()
    close = panel["close"]
    if split is None:
        split = three_way_split(close.index)
    is_mask, val_mask = split.is_mask, split.val_mask
    h = cfg.target_horizon
    ids = list(member_signals)
    signals = [member_signals[i] for i in ids]

    # ── axis 1: the set's own combined-model OOS IC ──
    pred = _combined_prediction(signals, close, is_mask, cfg, params.marginal_model)
    combined_is_ic, _ = (_pooled_ic(pred, close, h, is_mask, is_mask, is_mask)
                         if pred is not None else (None, 0))
    combined_val_ic, val_n = (_pooled_ic(pred, close, h, val_mask, is_mask, split.is_val_mask)
                              if pred is not None else (None, 0))

    # ── axis 2: internal participation ratio, normalised by set size ──
    flat = {
        i: _flat_z(member_signals[i], close,
                   stat_mask=split.is_val_mask, row_mask=split.is_val_mask)
        for i in ids
    }
    if len(ids) >= 2:
        corr = pd.DataFrame(flat).corr().to_numpy(dtype=float)
        pr = participation_ratio(corr)
        independence = (pr / len(ids)) if pr is not None else None
    else:
        independence = 1.0  # a singleton set is trivially non-redundant

    # ── axis 3: robustness of the COMBINED signal ──
    robust = (_robustness(pred, panel, cfg, split, params, combined_val_ic,
                          expected_sign, refit_signals=signals,
                          refit_base_signals=None,
                          standalone_overall_ic=combined_val_ic)
              if pred is not None else
              {"robustness": None, "cpcv_ic_mean": None, "cpcv_ic_std": None,
               "cpcv_n_folds": 0, "sign_consistency": None,
               "plateau_penalty": None, "jitter_ics": [],
               "cpcv_score_kind": "refit_combined",
               "cpcv_model": params.cpcv_model or params.marginal_model,
               "cpcv_fast": bool(params.cpcv_fast),
               "cpcv_with_ic_mean": None, "cpcv_with_ic_std": None,
               "cpcv_base_ic_mean": None, "cpcv_base_ic_std": None,
               "standalone_cpcv_ic_mean": None, "standalone_cpcv_ic_std": None,
               "standalone_cpcv_n_folds": 0})

    # ── regime axis: the set's own combined-model IC on the crash bars ──
    stress_mask = _stress_mask(close, params, split.is_val_mask)
    val_stress = np.asarray(val_mask, dtype=bool) & stress_mask
    if pred is not None and val_stress.any():
        stress_ic, n_stress = _pooled_ic(pred, close, h, val_stress, is_mask,
                                         split.is_val_mask)
        regime_independence = (stress_ic if (stress_ic is not None
                                             and n_stress >= params.regime_min_obs)
                               else None)
    else:
        stress_ic, n_stress, regime_independence = None, 0, None

    # ── axis 4: parsimony per member ──
    parsimony = None
    if member_codes:
        total = sum(complexity(member_codes.get(i, "")) for i in ids)
        parsimony = -float(total) / max(1, len(ids))

    # ── gates on the combined signal ──
    reasons: dict[str, str] = {}
    coverage = _coverage(pred, close, split.is_val_mask) if pred is not None else 0.0
    coverage_ok = coverage >= params.gate_coverage
    if not coverage_ok:
        reasons["coverage"] = f"{coverage:.3f} < τ={params.gate_coverage}"

    degradation_ok: bool | None
    deg_ratio = None
    if combined_is_ic is None or combined_val_ic is None \
            or abs(combined_is_ic) < params.min_is_ic:
        degradation_ok = None
    else:
        deg_ratio = (combined_val_ic * np.sign(combined_is_ic)) / abs(combined_is_ic)
        degradation_ok = bool(deg_ratio >= params.gate_degradation)
        if not degradation_ok:
            reasons["degradation"] = f"OOS/IS={deg_ratio:.3f} < τ={params.gate_degradation}"

    # DIAG only — deflation is a selection-time control (research_eval.publish, WS1),
    # not a per-candidate search gate.  ``deflation_ok`` stays ``None``.
    defl = deflation.deflated_ic(
        abs(combined_val_ic) if combined_val_ic is not None else None,
        int(val_n), params.n_trials)

    # economic-realism gate on the SET's combined signal (P5, WS3)
    tc = (_turnover_netcost(pred, close, h, split.is_val_mask, params)
          if pred is not None else
          {"turnover": None, "gross_ret": None, "net_ret": None, "net_gross_ratio": None})
    cost_ok: bool | None = None
    if params.gate_turnover is not None and tc["turnover"] is not None:
        cost_ok = bool(tc["turnover"] <= params.gate_turnover)
        if not cost_ok:
            reasons["cost"] = f"turnover={tc['turnover']:.3f} > τ={params.gate_turnover}"

    gates = GateResults(coverage_ok=coverage_ok, degradation_ok=degradation_ok,
                        deflation_ok=None, cost_ok=cost_ok, reasons=reasons)
    objective = ObjectiveVector(
        marginal_value=combined_val_ic,
        independence=independence,
        robustness=robust["robustness"],
        parsimony=parsimony,
        regime_independence=regime_independence,
    )
    diagnostics = {
        "set_members": ids,
        "set_size": len(ids),
        "combined_is_ic": combined_is_ic, "combined_val_ic": combined_val_ic,
        "val_n_obs": int(val_n),
        "internal_participation_ratio": (
            independence * len(ids) if independence is not None else None),
        "cpcv_ic_mean": robust["cpcv_ic_mean"], "cpcv_ic_std": robust["cpcv_ic_std"],
        "cpcv_n_folds": robust["cpcv_n_folds"],
        "cpcv_score_kind": robust["cpcv_score_kind"],
        "cpcv_model": robust["cpcv_model"],
        "cpcv_fast": robust["cpcv_fast"],
        "cpcv_with_ic_mean": robust["cpcv_with_ic_mean"],
        "cpcv_with_ic_std": robust["cpcv_with_ic_std"],
        "cpcv_base_ic_mean": robust["cpcv_base_ic_mean"],
        "cpcv_base_ic_std": robust["cpcv_base_ic_std"],
        "standalone_cpcv_ic_mean": robust["standalone_cpcv_ic_mean"],
        "standalone_cpcv_ic_std": robust["standalone_cpcv_ic_std"],
        "standalone_cpcv_n_folds": robust["standalone_cpcv_n_folds"],
        "sign_consistency": robust["sign_consistency"],
        "regime_independence": regime_independence,
        "regime_kind": params.regime_kind,
        "stress_ic_with": stress_ic, "n_stress_obs": int(n_stress),
        "perturbation_penalty": robust.get("perturbation_penalty"),
        "turnover": tc["turnover"], "net_ret": tc["net_ret"],
        "net_gross_ratio": tc["net_gross_ratio"],
        "coverage": coverage,
        "degradation_ratio": (float(deg_ratio) if deg_ratio is not None else None),
        "deflation": defl,
        "n_trials": params.n_trials,
    }
    behavior = (_behavior_descriptors(pred, close, h, split, params)
                if pred is not None
                else {"trend_reversal": None, "signal_speed": None,
                      "stress_activation": None})

    return FitnessResult(candidate_id=candidate_id, objective=objective, gates=gates,
                         diagnostics=diagnostics, behavior=behavior,
                         raw={"split_sizes": split.sizes})
