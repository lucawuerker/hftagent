"""Deterministic diagnostics → NL mutation brief for execution programs (E2).

The execution twin of the factor arm's ``reflection.py``: a **rule-based**
renderer that turns the exec harness's teacher-channel diagnostics into the
natural-language brief fed to the mutating LLM.  No LLM writes it — the prime
directive (the LLM never influences its own reward) holds; this is the reward
channel *explaining itself*, deterministically.
"""

from __future__ import annotations

from typing import Any

from quant_fund_agent.research_eval.fitness import FitnessResult


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x)
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def exec_mutation_brief(fitness: FitnessResult) -> str:
    """Render one executor's FitnessResult into the teacher brief."""
    d = fitness.diagnostics
    obj = fitness.objective
    lines: list[str] = []

    lines.append(
        f"EXECUTION FITNESS for `{fitness.candidate_id}` "
        f"({'PASSED all gates' if fitness.selectable else 'FAILED a gate'})")
    lines.append(
        f"- mean net VAL Sharpe (per bar) across {d.get('n_signals', '?')} frozen "
        f"signal(s): {_fmt(obj.marginal_value)} "
        f"(IS: {_fmt(d.get('mean_is_net_sharpe'))}, "
        f"dispersion across signals: {_fmt(d.get('val_sharpe_dispersion'))})")
    lines.append(
        f"- cost efficiency (net/gross capture): {_fmt(obj.robustness)} | "
        f"mean turnover/bar: {_fmt(d.get('mean_turnover'))} | "
        f"active on {_fmt(d.get('mean_activity'))} of bars")

    per_signal = d.get("per_signal") or []
    if per_signal:
        lines.append("- per-signal net VAL Sharpe: " + ", ".join(
            f"s{s.get('signal')}={_fmt(s.get('val_net_sharpe'))}"
            f"(capture {_fmt(s.get('capture'), 2)})" for s in per_signal))

    jit = [j for j in (d.get("jitter_val_sharpes") or []) if j is not None]
    if jit:
        lines.append(
            "- param-jitter probe (±10% params) net VAL Sharpes: "
            + ", ".join(_fmt(j) for j in jit))

    sens = per_signal[0] if per_signal else {}
    if "cost_x0.5" in sens or "cost_x1.5" in sens:
        lines.append(
            f"- cost sensitivity (signal 0): half costs → "
            f"{_fmt(sens.get('cost_x0.5'))}, 1.5× costs → "
            f"{_fmt(sens.get('cost_x1.5'))}")

    # ── rule-based advice (deterministic; ordered most→least severe) ──
    advice: list[str] = []
    reasons = fitness.gates.reasons or {}
    if "causality" in reasons:
        advice.append(
            "CAUSALITY FAILED: your weights at bar t change when the future "
            "changes. Remove any full-sample statistic (mean/std over the whole "
            "frame), centred windows or forward-looking ops; standardise with "
            "expanding or trailing windows only.")
    if "validity" in reasons:
        advice.append(
            f"OUTPUT CONTRACT VIOLATED ({reasons['validity']}): keep every "
            "weight finite, |w| within the per-name bound, gross within the "
            "leverage bound, and (cross-sectional regime) the book near "
            "dollar-neutral.")
    if "activity" in reasons:
        advice.append(
            "BOOK TOO FLAT: the program trades almost never. Loosen the entry "
            "threshold, or add a baseline exposure it modulates.")
    if "turnover" in reasons:
        advice.append(
            "TURNOVER OVER CEILING: add hysteresis (separate entry/exit bands), "
            "trade on a slower rebalance clock, or smooth the target book.")
    if "degradation" in reasons:
        advice.append(
            "IS→VAL DEGRADATION: the construction fits the IS window's "
            "personality. Prefer fewer, more universal rules over tuned "
            "constants.")
    cap = obj.robustness
    if cap is not None and cap < 0.5 and "turnover" not in reasons:
        advice.append(
            f"COST DRAG: only {_fmt(cap, 2)} of gross P&L survives costs — "
            "cut turnover (wider bands, slower clock, partial rebalancing "
            "toward the target instead of jumping to it).")
    disp = d.get("val_sharpe_dispersion")
    if disp is not None and obj.marginal_value is not None \
            and disp > abs(obj.marginal_value):
        advice.append(
            "SIGNAL CO-ADAPTATION: performance varies more across the frozen "
            "signals than its mean — condition on market state (vol, drawdown, "
            "signal age), never on one alpha's quirks.")
    if jit and obj.marginal_value is not None and any(
            j < 0.5 * obj.marginal_value for j in jit):
        advice.append(
            "KNIFE-EDGE PARAMS: a ±10% param nudge collapses the Sharpe — move "
            "to a plateau (rounder thresholds, less brittle interactions).")
    if not advice:
        advice.append(
            "All gates passed. Improve the dominated axes: raise net Sharpe via "
            "state-conditional sizing (vol targeting, drawdown de-risking), or "
            "cut cost drag while keeping the capture ratio.")

    lines.append("ADVICE:")
    lines.extend(f"  {i+1}. {a}" for i, a in enumerate(advice))
    return "\n".join(lines)
