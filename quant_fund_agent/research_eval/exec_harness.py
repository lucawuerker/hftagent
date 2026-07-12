"""Deterministic fitness harness for execution programs (the exec reward channel).

The execution twin of :mod:`research_eval.harness`: turns one candidate
**executor** (a compiled ``BaseExecutor``) into a
:class:`~quant_fund_agent.research_eval.fitness.FitnessResult` by running it
over the K **frozen evaluation signals** (``execution.signal_freeze``) on the
dev-sliced panel, through a cost-aware book simulation.

`ObjectiveVector` slot mapping (see ``docs/execution-evolution/DESIGN.md``
§Fitness — the dataclasses are reused verbatim; honest names live here and in
``diagnostics``):

* ``marginal_value``       := mean net-of-cost **VAL** Sharpe across the K
                              signals (per-bar units) — the primary axis.
* ``independence``         := cross-signal generalisation
                              ``mean − λ·dispersion`` of the K VAL Sharpes.
* ``robustness``           := cost efficiency — mean net÷gross capture ratio.
* ``parsimony``            := −AST complexity of the program source.
* ``structural_novelty``   := min normalised code-edit distance to the nearest
                              **archived executor** (same metric as the factor
                              axis, reusing ``harness._structural_novelty``);
                              ``None`` when the archive is empty.

Gates (slot → meaning):

* ``coverage_ok``     := output-contract validity + min-activity floor.
* ``degradation_ok``  := IS→VAL net-Sharpe ratio ≥ τ with matching sign.
* ``deflation_ok``    := deflated-Sharpe probability > 0.5 at the (family)
                         ``n_trials`` — evaluated only when
                         ``selection_deflation="on"`` (mirrors WS1: deflation
                         is a *publish*-time control by default).
* ``cost_ok``         := truncation-replay **causality probe** passes AND
                         (when a ceiling is set) turnover ≤ τ_turn.

Conventions (leak-free, inherited from the factor harness):

* The panel/signals arriving here are **dev-sliced** (TEST physically absent —
  the service guarantees it); the split is dev-relative.
* Book P&L is ``weights.shift(1) × 1-bar forward return`` — identical to
  ``strategy_backtester._portfolio_returns`` (positions formed at bar *t*
  trade from *t+1*).  The last dev bar's forward return needs a TEST price and
  is NaN on the dev slice, so it drops out of every candidate's score
  uniformly (DESIGN §Leak-free, point 2).
* Costs: ``cost_rate × Σ|Δw|`` charged on the bar the trade is placed.
* ``NaN`` weights = "no position" (0); ``±inf`` / bound breaches fail validity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from quant_fund_agent.execution.base import (
    DEFAULT_MAX_GROSS,
    DEFAULT_MAX_NAME,
    DEFAULT_NEUTRAL_TOL,
    BaseExecutor,
    run_executor,
    validate_weights,
)
from quant_fund_agent.execution.state import build_state_frames
from quant_fund_agent.research_eval import deflation
from quant_fund_agent.research_eval.fitness import (
    FitnessResult,
    GateResults,
    ObjectiveVector,
    complexity,
)
from quant_fund_agent.research_eval.splits import ThreeWaySplit, three_way_split

log = logging.getLogger("research_eval.exec_harness")


@dataclass
class ExecEvalParams:
    """Deterministic scoring knobs — held identical across every candidate."""

    n_trials: int = 1                    # executor-family trial count (deflation)
    cost_rate: float = 5e-4              # ≈5 bps per unit turnover
    lambda_dispersion: float = 0.5       # cross-signal axis: mean − λ·std
    gate_turnover: float | None = None   # per-bar mean Σ|Δw| ceiling (None → not gated)
    gate_degradation: float = 0.5        # τ_deg: min VAL/IS net-Sharpe ratio (same sign)
    min_is_sharpe: float = 0.005         # |IS per-bar Sharpe| below → degradation not evaluable
    min_activity: float = 0.05           # min fraction of dev bars with a non-zero book
    max_gross: float = DEFAULT_MAX_GROSS
    max_name: float = DEFAULT_MAX_NAME
    neutral_tol: float = DEFAULT_NEUTRAL_TOL
    selection_deflation: str = "off"     # "off" (publish-time control, WS1) | "on"
    causality_probe_points: int = 2      # truncation-replay probes per evaluation
    cost_sensitivity: bool = True        # ±50% cost re-score (diagnostic)
    vol_window: int = 20
    adv_window: int = 20


# ── book simulation ────────────────────────────────────────────────────────────

def _book_pnl(weights: pd.DataFrame, close: pd.DataFrame,
              cost_rate: float) -> dict[str, pd.Series]:
    """Per-bar gross/net book returns + turnover from a weight frame.

    ``gross[t] = Σ_i w[t-1, i] · (close[t+1]/close[t] − 1)`` (the deployed
    convention); ``turnover[t] = Σ_i |w[t,i] − w[t-1,i]|``;
    ``net[t] = gross[t] − cost_rate · turnover[t]``.  NaN weights are 0.
    """
    from quant_fund_agent.backtesting.data_loader import forward_returns

    w = weights.fillna(0.0)
    fwd1 = forward_returns(close, horizon=1).reindex(index=w.index, columns=w.columns)
    gross = (w.shift(1) * fwd1).sum(axis=1, min_count=1)
    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[:1] = w.iloc[:1].abs().sum(axis=1)  # entering the initial book costs too
    net = gross - cost_rate * turnover
    return {"gross": gross, "net": net, "turnover": turnover,
            "gross_exposure": w.abs().sum(axis=1)}


def _sharpe(x: pd.Series, mask: np.ndarray) -> tuple[float | None, int]:
    """Per-bar (non-annualised) Sharpe of ``x`` over ``mask`` rows; (value, n)."""
    v = x.to_numpy(dtype=float)[np.asarray(mask, dtype=bool)]
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return None, len(v)
    sd = float(np.std(v, ddof=1))
    if sd == 0:
        return None, len(v)
    return float(np.mean(v) / sd), len(v)


# ── the causality probe (truncation-replay) ────────────────────────────────────

def causality_probe(
    executor: BaseExecutor,
    signal: pd.DataFrame,
    panel: dict[str, Any],
    *,
    n_points: int = 2,
    vol_window: int = 20,
    adv_window: int = 20,
) -> dict[str, Any]:
    """Prove the program's weights at bar ``t`` ignore everything after ``t``.

    Re-runs the executor on the panel/signal **physically truncated** at each
    probe point and compares the weight row at ``t`` with the full run's row.
    Truncation-replay (not value perturbation) cannot be dodged by
    perturbation-insensitive ops — full-sample standardisation, future-window
    ops and any other look-ahead move the truncated row.  Probe points sit at
    the interior thirds of the window so warm-up NaNs don't mask a leak.
    """
    n = len(signal.index)
    if n < 12:
        return {"passed": None, "points": [], "reason": "window too short to probe"}
    points = sorted({max(4, (i + 1) * n // (n_points + 1)) for i in range(n_points)})

    state_full = build_state_frames(panel, signal, vol_window=vol_window,
                                    adv_window=adv_window)
    w_full = run_executor(executor, signal, state_full, panel["close"]).fillna(0.0)

    results = []
    for t in points:
        sig_t = signal.iloc[: t + 1]
        panel_t = {k: df.iloc[: t + 1] for k, df in panel.items()}
        state_t = build_state_frames(panel_t, sig_t, vol_window=vol_window,
                                     adv_window=adv_window)
        w_t = run_executor(executor, sig_t, state_t, panel_t["close"]).fillna(0.0)
        a = w_full.iloc[t].to_numpy(dtype=float)
        b = w_t.iloc[-1].to_numpy(dtype=float)
        diff = float(np.max(np.abs(a - b))) if len(a) else 0.0
        results.append({"t": int(t), "max_diff": diff, "ok": bool(diff <= 1e-10)})
    return {"passed": all(r["ok"] for r in results), "points": results}


# ── the orchestrator ───────────────────────────────────────────────────────────

def evaluate_executor(
    executor: BaseExecutor,
    frozen_signals: Sequence[pd.DataFrame],
    panel: dict[str, Any],
    split: ThreeWaySplit | None = None,
    *,
    params: ExecEvalParams | None = None,
    candidate_code: str | None = None,
    candidate_id: str = "executor",
    jitter_executors: Sequence[BaseExecutor] | None = None,
    archive_codes: Sequence[str] | None = None,
) -> FitnessResult:
    """Score one execution program against the K frozen signals → FitnessResult.

    ``panel`` and every frame in ``frozen_signals`` must already be dev-sliced
    (IS∪VAL only); ``split`` is dev-relative (``None`` derives 75/25 IS/VAL
    over the dev window with an empty TEST).  ``jitter_executors``
    (param-jittered variants, the plateau probe) are scored on their mean VAL
    net Sharpe as a *diagnostic* — the axes are unchanged.  ``archive_codes``
    (sources of the archived executors) feed the structural-novelty axis.
    """
    params = params or ExecEvalParams()
    close = panel["close"]
    if split is None:
        split = three_way_split(close.index, is_frac=0.75, val_frac=0.249)
    is_mask = np.asarray(split.is_mask, dtype=bool)
    val_mask = np.asarray(split.val_mask, dtype=bool)
    dev_mask = np.asarray(split.is_val_mask, dtype=bool)

    reasons: dict[str, str] = {}
    per_signal: list[dict[str, Any]] = []
    validity_problems: list[str] = []
    val_net_by_signal: list[pd.Series] = []

    for k, sig in enumerate(frozen_signals):
        sig = sig.reindex(index=close.index, columns=close.columns)
        state = build_state_frames(panel, sig, vol_window=params.vol_window,
                                   adv_window=params.adv_window)
        try:
            weights = run_executor(executor, sig, state, close)
        except Exception as e:  # noqa: BLE001 — a crashing program is a scored failure
            reasons["validity"] = f"executor raised on signal {k}: {e}"
            gates = GateResults(coverage_ok=False, reasons=reasons)
            return FitnessResult(candidate_id=candidate_id,
                                 objective=ObjectiveVector(),
                                 gates=gates,
                                 diagnostics={"error": reasons["validity"]})

        problems = validate_weights(weights, executor.regime,
                                    max_gross=params.max_gross,
                                    max_name=params.max_name,
                                    neutral_tol=params.neutral_tol)
        validity_problems.extend(f"signal {k}: {p}" for p in problems)

        pnl = _book_pnl(weights, close, params.cost_rate)
        is_sr, _ = _sharpe(pnl["net"], is_mask)
        val_sr, val_n = _sharpe(pnl["net"], val_mask)
        val_gross_sr, _ = _sharpe(pnl["gross"], val_mask)

        vg = pnl["gross"].to_numpy(dtype=float)[val_mask]
        vn = pnl["net"].to_numpy(dtype=float)[val_mask]
        fin = np.isfinite(vg) & np.isfinite(vn)
        gross_mean = float(np.mean(vg[fin])) if fin.any() else None
        net_mean = float(np.mean(vn[fin])) if fin.any() else None
        capture = (net_mean / gross_mean
                   if gross_mean is not None and gross_mean > 0 else None)

        turnover_dev = pnl["turnover"].to_numpy(dtype=float)[dev_mask]
        turnover_mean = (float(np.mean(turnover_dev[np.isfinite(turnover_dev)]))
                         if np.isfinite(turnover_dev).any() else None)
        active = pnl["gross_exposure"].to_numpy(dtype=float)[dev_mask]
        activity = float(np.mean(active > 1e-12)) if len(active) else 0.0

        # ±50% cost sensitivity — the cost-model-gaming diagnostic
        sens = {}
        if params.cost_sensitivity:
            for lbl, rate in (("cost_x0.5", 0.5 * params.cost_rate),
                              ("cost_x1.5", 1.5 * params.cost_rate)):
                alt = _book_pnl(weights, close, rate)
                sens[lbl], _ = _sharpe(alt["net"], val_mask)

        per_signal.append({
            "signal": k, "is_net_sharpe": is_sr, "val_net_sharpe": val_sr,
            "val_gross_sharpe": val_gross_sr, "val_n_obs": val_n,
            "net_mean": net_mean, "gross_mean": gross_mean, "capture": capture,
            "turnover": turnover_mean, "activity": activity, **sens,
        })
        val_net_by_signal.append(pnl["net"])

    # ── aggregate across the K signals ──
    val_srs = [s["val_net_sharpe"] for s in per_signal if s["val_net_sharpe"] is not None]
    is_srs = [s["is_net_sharpe"] for s in per_signal if s["is_net_sharpe"] is not None]
    captures = [s["capture"] for s in per_signal if s["capture"] is not None]
    turnovers = [s["turnover"] for s in per_signal if s["turnover"] is not None]
    activities = [s["activity"] for s in per_signal]

    mean_val_sr = float(np.mean(val_srs)) if val_srs else None
    disp_val_sr = float(np.std(val_srs)) if len(val_srs) >= 2 else 0.0
    mean_is_sr = float(np.mean(is_srs)) if is_srs else None
    generalisation = (mean_val_sr - params.lambda_dispersion * disp_val_sr
                      if mean_val_sr is not None else None)
    mean_capture = float(np.mean(captures)) if captures else None
    mean_turnover = float(np.mean(turnovers)) if turnovers else None
    mean_activity = float(np.mean(activities)) if activities else 0.0

    # equal-weight portfolio of the K books → the deflation statistic
    dsr_prob = None
    if val_net_by_signal:
        avg_net = pd.concat(val_net_by_signal, axis=1).mean(axis=1, skipna=False)
        avg_sr, avg_n = _sharpe(avg_net, val_mask)
        if avg_sr is not None:
            sr_var = float(np.var(val_srs)) if len(val_srs) >= 2 else 0.0
            dsr_prob = deflation.deflated_sharpe_ratio(
                avg_sr, n_obs=avg_n, sr_variance=sr_var,
                n_trials=max(1, int(params.n_trials)))

    # ── plateau probe (diagnostic): param-jittered variants ──
    jitter_srs: list[float | None] = []
    for j in (jitter_executors or []):
        try:
            sig0 = frozen_signals[0].reindex(index=close.index, columns=close.columns)
            st = build_state_frames(panel, sig0, vol_window=params.vol_window,
                                    adv_window=params.adv_window)
            wj = run_executor(j, sig0, st, close)
            sr_j, _ = _sharpe(_book_pnl(wj, close, params.cost_rate)["net"], val_mask)
            jitter_srs.append(sr_j)
        except Exception:  # noqa: BLE001 — a broken probe is a data point, not a crash
            jitter_srs.append(None)

    # ── causality probe (truncation-replay) — the leak gate ──
    probe = causality_probe(executor, frozen_signals[0].reindex(
        index=close.index, columns=close.columns), panel,
        n_points=params.causality_probe_points,
        vol_window=params.vol_window, adv_window=params.adv_window)
    causality_ok = probe["passed"]
    if causality_ok is False:
        reasons["causality"] = f"truncation-replay mismatch: {probe['points']}"

    # ── gates ──
    validity_ok = not validity_problems
    if validity_problems:
        reasons["validity"] = "; ".join(validity_problems[:3])
    activity_ok = mean_activity >= params.min_activity
    if not activity_ok:
        reasons["activity"] = (f"active on {mean_activity:.3f} of dev bars "
                               f"< floor {params.min_activity}")
    coverage_ok = validity_ok and activity_ok

    degradation_ok: bool | None
    deg_ratio = None
    if mean_is_sr is None or mean_val_sr is None or abs(mean_is_sr) < params.min_is_sharpe:
        degradation_ok = None
    else:
        deg_ratio = (mean_val_sr * np.sign(mean_is_sr)) / abs(mean_is_sr)
        degradation_ok = bool(deg_ratio >= params.gate_degradation)
        if not degradation_ok:
            reasons["degradation"] = (f"VAL/IS={deg_ratio:.3f} "
                                      f"< τ={params.gate_degradation}")

    deflation_ok: bool | None = None
    if params.selection_deflation == "on":
        deflation_ok = bool(dsr_prob is not None and dsr_prob > 0.5)
        if not deflation_ok:
            reasons["deflation"] = f"DSR prob={dsr_prob} ≤ 0.5 at n_trials={params.n_trials}"

    turnover_ok = True
    if params.gate_turnover is not None and mean_turnover is not None:
        turnover_ok = mean_turnover <= params.gate_turnover
        if not turnover_ok:
            reasons["turnover"] = (f"turnover={mean_turnover:.3f} "
                                   f"> τ={params.gate_turnover}")
    cost_ok: bool | None
    if causality_ok is None:
        cost_ok = None if turnover_ok else False
    else:
        cost_ok = bool(causality_ok and turnover_ok)

    gates = GateResults(coverage_ok=coverage_ok, degradation_ok=degradation_ok,
                        deflation_ok=deflation_ok, cost_ok=cost_ok, reasons=reasons)

    # structural novelty vs the archived executors — the SAME metric as the
    # factor axis (min normalised code-edit distance), reused so the two arms
    # cannot drift.
    from quant_fund_agent.research_eval.harness import _structural_novelty

    novelty = _structural_novelty(candidate_code, list(archive_codes or []))

    objective = ObjectiveVector(
        marginal_value=mean_val_sr,
        independence=generalisation,
        robustness=mean_capture,
        parsimony=(-float(complexity(candidate_code))
                   if candidate_code is not None else None),
        structural_novelty=novelty["structural_novelty"],
    )

    diagnostics = {
        "per_signal": per_signal,
        "n_signals": len(frozen_signals),
        "mean_val_net_sharpe": mean_val_sr,
        "mean_is_net_sharpe": mean_is_sr,
        "val_sharpe_dispersion": disp_val_sr,
        "cross_signal_generalisation": generalisation,
        "mean_capture": mean_capture,
        "mean_turnover": mean_turnover,
        "mean_activity": mean_activity,
        "degradation_ratio": deg_ratio,
        "deflated_sharpe_prob": dsr_prob,
        "n_trials": params.n_trials,
        "selection_deflation": params.selection_deflation,
        "causality_probe": probe,
        "validity_problems": validity_problems,
        "jitter_val_sharpes": jitter_srs,
        "complexity": (complexity(candidate_code)
                       if candidate_code is not None else None),
        "structural_novelty": novelty["structural_novelty"],
        "novelty_min_book_distance": novelty.get("novelty_min_book_distance"),
        "regime": executor.regime,
        "cost_rate": params.cost_rate,
    }

    return FitnessResult(candidate_id=candidate_id, objective=objective,
                         gates=gates, diagnostics=diagnostics,
                         raw={"split_sizes": split.sizes})
