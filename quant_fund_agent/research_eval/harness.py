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
* Family 3 (independence, ``[CORE]``): Δ participation ratio − soft max-|corr| penalty.
* Family 4 (robustness, ``[CORE]``): ``mean_cpcv(IC) − λ·std_cpcv(IC) + sign_bonus``;
  the OOS/IS degradation and deflated-IC hard gates.
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
from dataclasses import dataclass, field
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
    marginal_model: str = "ridge"     # estimator combining factors for the LOCO axis
    plateau_weight: float = 1.0       # weight of the window-jitter plateau penalty
    cpcv_groups: int = 6
    cpcv_k: int = 2
    embargo: int = 0
    gate_coverage: float = 0.5        # τ_cov: min non-NaN (date,ticker) fraction
    gate_degradation: float = 0.5     # τ_deg: min OOS/IS IC ratio (same sign)
    min_is_ic: float = 0.005          # |IS IC| below this → degradation gate not evaluated
    ic_decay_horizons: tuple[int, ...] = (1, 3, 6, 12, 24)


# ── low-level IC / prediction helpers (signal-based, leak-aware) ──────────────

def _pooled_ic(
    sig: pd.DataFrame, close: pd.DataFrame, horizon: int,
    row_mask: np.ndarray | None = None, stat_mask: np.ndarray | None = None,
) -> tuple[float | None, int]:
    """Pooled per-underlying time-series Spearman IC of ``sig`` vs its forward return.

    ``row_mask`` selects the rows scored; ``stat_mask`` selects the rows whose stats
    standardise the signal (pass the IS window to keep an OOS score leak-free).  The
    forward return is always taken from the *full* ``close`` so a label at the end of
    a block still uses the true future price.
    """
    s = sig.reindex(index=close.index, columns=close.columns)
    stat_idx = close.index[stat_mask] if stat_mask is not None else None
    z = per_underlying_zscore(s, stat_idx)
    x = z.to_numpy(dtype=float)
    y = forward_returns(close, horizon=horizon).to_numpy(dtype=float)
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

    is_rows = np.repeat(np.asarray(is_mask), n_cols)
    train = np.flatnonzero(is_rows & np.isfinite(y))
    if len(train) < 30:
        return None
    fast = cfg.fast_model_params(model) if hasattr(cfg, "fast_model_params") else None
    est = build_estimator(model, fast)
    est.fit(X[train], y[train])
    pred = np.asarray(est.predict(X), dtype=float).reshape(n_rows, n_cols)
    return pd.DataFrame(pred, index=index, columns=cols)


def _flat_z(sig: pd.DataFrame, close: pd.DataFrame) -> np.ndarray:
    """Per-underlying z-scored signal, reindexed to the panel grid and flattened."""
    s = sig.reindex(index=close.index, columns=close.columns)
    return per_underlying_zscore(s).to_numpy(dtype=float).ravel()


# ── the axes ──────────────────────────────────────────────────────────────────

def _marginal_value(
    candidate: pd.DataFrame, book: Sequence[pd.DataFrame], panel, cfg,
    is_mask, val_mask, model: str,
) -> dict[str, Any]:
    """LOCO: ΔOOS-IC of the combined model from adding the candidate to the book."""
    close = panel["close"]
    h = cfg.target_horizon

    with_pred = _combined_prediction([*book, candidate], close, is_mask, cfg, model)
    with_ic = _pooled_ic(with_pred, close, h, val_mask, is_mask)[0] if with_pred is not None else None

    if book:
        base_pred = _combined_prediction(list(book), close, is_mask, cfg, model)
        base_ic = _pooled_ic(base_pred, close, h, val_mask, is_mask)[0] if base_pred is not None else None
    else:
        base_ic = 0.0  # empty book → the candidate is the whole edge

    if with_ic is None:
        return {"marginal_value": None, "with_ic": None, "base_ic": base_ic}
    # The combined model predicts the forward return, so a useful signal has a
    # POSITIVE IC; ΔIC then measures the edge the candidate adds on top of the book.
    marginal = with_ic - (base_ic if base_ic is not None else 0.0)
    return {"marginal_value": marginal, "with_ic": with_ic, "base_ic": base_ic}


def _residual_ic(candidate: pd.DataFrame, book: Sequence[pd.DataFrame], panel, cfg) -> float | None:
    """Orthogonalised (residual) IC — the candidate's edge beyond the book's span.

    Regress the flattened candidate on the flattened book signals (OLS over their
    finite rows), then Spearman-correlate the residual with the forward return.
    Diagnostic (Family 2); with an empty book it is the standalone pooled IC.
    """
    close = panel["close"]
    y_fwd = forward_returns(close, horizon=cfg.target_horizon).to_numpy(dtype=float).ravel()
    cand = _flat_z(candidate, close)

    if not book:
        resid = cand
    else:
        B = np.column_stack([_flat_z(s, close) for s in book])
        finite = np.isfinite(cand) & np.isfinite(B).all(axis=1)
        if finite.sum() < 30:
            return None
        Bf = np.column_stack([np.ones(finite.sum()), B[finite]])
        beta, *_ = np.linalg.lstsq(Bf, cand[finite], rcond=None)
        resid = np.full_like(cand, np.nan)
        resid[finite] = cand[finite] - Bf @ beta
    return _spearman(resid, y_fwd)[0]


def _independence(candidate: pd.DataFrame, book: Sequence[pd.DataFrame], panel, cfg,
                  corr_penalty: float) -> dict[str, Any]:
    """Δ participation ratio (book vs book+candidate) minus a soft max-|corr| penalty."""
    close = panel["close"]
    cand = _flat_z(candidate, close)

    if not book:
        return {"independence": 1.0, "delta_participation": 1.0, "max_abs_corr": 0.0,
                "pr_before": 0.0, "pr_after": 1.0}

    cols = {f"b{i}": _flat_z(s, close) for i, s in enumerate(book)}
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


def _robustness(candidate: pd.DataFrame, panel, cfg, split: ThreeWaySplit, params: EvalParams,
                overall_ic: float | None, expected_sign: int | None,
                jitter_signals: Sequence[pd.DataFrame] | None = None) -> dict[str, Any]:
    """CPCV IC distribution → ``mean − λ·std − plateau`` (sign-aligned) + sign bonus."""
    close = panel["close"]
    h = cfg.target_horizon
    sign = 1.0 if (overall_ic is None or overall_ic >= 0) else -1.0

    folds = cpcv_folds(close.index, n_groups=params.cpcv_groups, k_test=params.cpcv_k,
                       horizon=h, embargo=params.embargo, mask=split.is_val_mask)
    ics: list[float] = []
    for fold in folds:
        ic, _ = _pooled_ic(candidate, close, h, fold.test, fold.test)
        if ic is not None:
            ics.append(ic * sign)  # align to the candidate's natural direction

    if len(ics) < 2:
        cpcv_mean = cpcv_std = None
        robustness = None
    else:
        cpcv_mean = float(np.mean(ics))
        cpcv_std = float(np.std(ics))
        robustness = cpcv_mean - params.lambda_std * cpcv_std

    # Parameter-sensitivity plateau test: score each window-jittered variant on
    # VAL exactly like the candidate; the penalty is the (sign-aligned) IC drop
    # from the candidate to the jitter mean, floored at 0 — a knife-edge factor
    # loses its edge under a ±10% window change, a plateau factor keeps it.
    plateau_penalty = None
    jitter_ics: list[float] = []
    if jitter_signals and overall_ic is not None:
        for jsig in jitter_signals:
            jic = _pooled_ic(jsig, close, h, split.val_mask, split.is_mask)[0]
            if jic is not None:
                jitter_ics.append(jic * sign)
        if jitter_ics:
            plateau_penalty = max(0.0, float(overall_ic * sign - np.mean(jitter_ics)))
            if robustness is not None:
                robustness -= params.plateau_weight * plateau_penalty

    sign_consistency = None
    if expected_sign is not None and overall_ic is not None:
        realized = 1 if overall_ic >= 0 else -1
        sign_consistency = bool(realized == int(np.sign(expected_sign) or 1))
        if robustness is not None:
            robustness += params.sign_bonus if sign_consistency else -params.sign_bonus

    return {"robustness": robustness, "cpcv_ic_mean": cpcv_mean, "cpcv_ic_std": cpcv_std,
            "cpcv_n_folds": len(ics), "sign_consistency": sign_consistency,
            "plateau_penalty": plateau_penalty, "jitter_ics": jitter_ics}


def _coverage(sig: pd.DataFrame, close: pd.DataFrame, mask: np.ndarray) -> float:
    """Fraction of (date,ticker) cells in ``mask`` where the signal is finite."""
    s = sig.reindex(index=close.index, columns=close.columns).to_numpy(dtype=float)[mask]
    if s.size == 0:
        return 0.0
    return float(np.isfinite(s).mean())


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
        ic_decay[str(hz)] = _pooled_ic(candidate_signal, close, hz, val_mask, is_mask)[0]
    standalone_ic = ic_decay.get(str(h))
    if standalone_ic is None:
        standalone_ic = _pooled_ic(candidate_signal, close, h, val_mask, is_mask)[0]

    is_ic, is_n = _pooled_ic(candidate_signal, close, h, is_mask, is_mask)
    val_ic, val_n = _pooled_ic(candidate_signal, close, h, val_mask, is_mask)

    # ── Family 2 — marginal / incremental value (CORE, primary) ──
    marg = _marginal_value(candidate_signal, book, panel, cfg, is_mask, val_mask,
                           params.marginal_model)
    residual_ic = _residual_ic(candidate_signal, book, panel, cfg)

    # ── Family 3 — independence (CORE) ──
    indep = _independence(candidate_signal, book, panel, cfg, params.corr_penalty)

    # ── Family 4 — robustness (CORE) ──
    robust = _robustness(candidate_signal, panel, cfg, split, params, val_ic,
                         expected_sign, jitter_signals)

    # ── parsimony axis ──
    parsimony = -float(complexity(candidate_code)) if candidate_code is not None else None

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

    # Deflated-IC t-stat > 0 for the current N_trials (the multiple-testing gate).
    defl = deflation.deflated_ic(abs(val_ic) if val_ic is not None else None,
                                 int(val_n), params.n_trials)
    deflation_ok: bool | None
    if defl["deflated_t"] is None:
        deflation_ok = None
    else:
        deflation_ok = bool(defl["deflated_t"] > 0)
        if not deflation_ok:
            reasons["deflation"] = f"deflated_t={defl['deflated_t']} ≤ 0 (N_trials={params.n_trials})"

    gates = GateResults(coverage_ok=coverage_ok, degradation_ok=degradation_ok,
                        deflation_ok=deflation_ok, cost_ok=None, reasons=reasons)

    objective = ObjectiveVector(
        marginal_value=marg["marginal_value"],
        independence=indep["independence"],
        robustness=robust["robustness"],
        parsimony=parsimony,
    )

    diagnostics = {
        "standalone_ic": standalone_ic,
        "ic_decay": ic_decay,
        "is_ic": is_ic, "val_ic": val_ic, "val_n_obs": int(val_n),
        "residual_ic": residual_ic,
        "with_ic": marg["with_ic"], "base_ic": marg["base_ic"],
        "max_abs_corr": indep["max_abs_corr"],
        "delta_participation": indep["delta_participation"],
        "pr_before": indep["pr_before"], "pr_after": indep["pr_after"],
        "cpcv_ic_mean": robust["cpcv_ic_mean"], "cpcv_ic_std": robust["cpcv_ic_std"],
        "cpcv_n_folds": robust["cpcv_n_folds"],
        "sign_consistency": robust["sign_consistency"],
        "plateau_penalty": robust["plateau_penalty"],
        "jitter_ics": robust["jitter_ics"],
        "coverage": coverage,
        "degradation_ratio": (float(deg_ratio) if deg_ratio is not None else None),
        "deflation": defl,
        "n_trials": params.n_trials,
        "complexity": (complexity(candidate_code) if candidate_code is not None else None),
    }

    return FitnessResult(candidate_id=candidate_id, objective=objective, gates=gates,
                         diagnostics=diagnostics, raw={"split_sizes": split.sizes})


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
    combined_is_ic, _ = (_pooled_ic(pred, close, h, is_mask, is_mask)
                         if pred is not None else (None, 0))
    combined_val_ic, val_n = (_pooled_ic(pred, close, h, val_mask, is_mask)
                              if pred is not None else (None, 0))

    # ── axis 2: internal participation ratio, normalised by set size ──
    flat = {i: _flat_z(member_signals[i], close) for i in ids}
    if len(ids) >= 2:
        corr = pd.DataFrame(flat).corr().to_numpy(dtype=float)
        pr = participation_ratio(corr)
        independence = (pr / len(ids)) if pr is not None else None
    else:
        independence = 1.0  # a singleton set is trivially non-redundant

    # ── axis 3: robustness of the COMBINED signal ──
    robust = (_robustness(pred, panel, cfg, split, params, combined_val_ic,
                          expected_sign)
              if pred is not None else
              {"robustness": None, "cpcv_ic_mean": None, "cpcv_ic_std": None,
               "cpcv_n_folds": 0, "sign_consistency": None,
               "plateau_penalty": None, "jitter_ics": []})

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

    defl = deflation.deflated_ic(
        abs(combined_val_ic) if combined_val_ic is not None else None,
        int(val_n), params.n_trials)
    deflation_ok: bool | None
    if defl["deflated_t"] is None:
        deflation_ok = None
    else:
        deflation_ok = bool(defl["deflated_t"] > 0)
        if not deflation_ok:
            reasons["deflation"] = (f"deflated_t={defl['deflated_t']} ≤ 0 "
                                    f"(N_trials={params.n_trials})")

    gates = GateResults(coverage_ok=coverage_ok, degradation_ok=degradation_ok,
                        deflation_ok=deflation_ok, cost_ok=None, reasons=reasons)
    objective = ObjectiveVector(
        marginal_value=combined_val_ic,
        independence=independence,
        robustness=robust["robustness"],
        parsimony=parsimony,
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
        "sign_consistency": robust["sign_consistency"],
        "coverage": coverage,
        "degradation_ratio": (float(deg_ratio) if deg_ratio is not None else None),
        "deflation": defl,
        "n_trials": params.n_trials,
    }
    return FitnessResult(candidate_id=candidate_id, objective=objective, gates=gates,
                         diagnostics=diagnostics, raw={"split_sizes": split.sizes})
