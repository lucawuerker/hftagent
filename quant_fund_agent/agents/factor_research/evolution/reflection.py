"""Reflection: the deterministic diagnostics dict → a natural-language mutation brief.

This is the **teacher channel** of the two-channel split: rich, specific,
even opinionated prose the LLM reads when mutating a parent.  It is rendered
*deterministically* from the harness diagnostics — no LLM writes it — so it can
never drift into flattery or self-reinforcement, and identical diagnostics
always produce the identical brief (reproducible prompts, cacheable runs).

The numbers quoted are all out-of-sample statistics (VAL-window ICs, CPCV fold
dispersion, correlations), never raw labels — the LLM sees *how* the candidate
failed, not the data it failed on, so the reward stays un-gameable.
"""

from __future__ import annotations

from typing import Any

from quant_fund_agent.research_eval.fitness import FitnessResult


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:+.{nd}f}"
    return str(x)


def _decay_line(ic_decay: dict[str, float | None] | None) -> str | None:
    if not ic_decay:
        return None
    cells = []
    best_h, best_ic = None, None
    for h, ic in ic_decay.items():
        cells.append(f"{h}b={_fmt(ic)}")
        if ic is not None and (best_ic is None or abs(ic) > abs(best_ic)):
            best_h, best_ic = h, ic
    line = "IC decay across horizons: " + ", ".join(cells) + "."
    if best_h is not None:
        line += f"  |IC| peaks at {best_h} bars ({_fmt(best_ic)})."
    return line


def mutation_brief(fitness: FitnessResult, *, book_size: int | None = None) -> str:
    """Render one candidate's dashboard into the NL brief for the mutating LLM.

    Sections: verdict (gates), the four Pareto axes with their raw ingredients,
    then targeted, rule-based *advice lines* — the "your edge lived only in
    2020 H1"-style observations the design calls for, derived purely from the
    numbers.  Verbose on purpose; the reward channel stays numeric elsewhere.
    """
    d = fitness.diagnostics
    obj = fitness.objective
    lines: list[str] = []

    # ── verdict ──
    if fitness.gates.passed:
        lines.append("VERDICT: passed every hard gate; competing on the Pareto front.")
    else:
        lines.append("VERDICT: FAILED hard gate(s) — "
                     + "; ".join(f"{k}: {v}" for k, v in fitness.gates.reasons.items())
                     + ".")

    # ── the axes ──
    lines.append(
        f"Marginal value (primary axis): adding this factor moved the combined "
        f"model's OOS IC by {_fmt(obj.marginal_value)} "
        f"(with={_fmt(d.get('with_ic'))}, book-only={_fmt(d.get('base_ic'))}"
        + (f", book size={book_size}" if book_size is not None else "") + ").")
    lines.append(
        f"Standalone IC at the target horizon: {_fmt(d.get('standalone_ic'))} "
        f"(IS={_fmt(d.get('is_ic'))}, VAL={_fmt(d.get('val_ic'))}).")
    decay = _decay_line(d.get("ic_decay"))
    if decay:
        lines.append(decay)
    metric = d.get("independence_metric", "residual_ic")
    lines.append(
        f"Independence (axis = {metric}): residual IC after regressing out the "
        f"book {_fmt(d.get('residual_ic'))} — the edge the book does NOT already "
        f"span; Δ participation ratio {_fmt(d.get('delta_participation'))} (book "
        f"effective factors {_fmt(d.get('pr_before'), 2)} → "
        f"{_fmt(d.get('pr_after'), 2)}); max |corr| to an existing factor: "
        f"{_fmt(d.get('max_abs_corr'))}.")
    lines.append(
        f"Robustness: CPCV IC mean {_fmt(d.get('cpcv_ic_mean'))} ± "
        f"{_fmt(d.get('cpcv_ic_std'))} over {d.get('cpcv_n_folds', 0)} purged "
        f"folds; OOS/IS degradation ratio {_fmt(d.get('degradation_ratio'))}.")
    if d.get("regime_independence") is not None or d.get("stress_ic_with") is not None:
        lines.append(
            f"Regime (crash-complementarity): marginal ΔIC on the "
            f"{d.get('regime_kind', 'drawdown')} stress bars "
            f"{_fmt(d.get('regime_independence'))} "
            f"(book-with={_fmt(d.get('stress_ic_with'))}, "
            f"book-only={_fmt(d.get('stress_ic_base'))}, "
            f"n_stress={d.get('n_stress_obs', 0)}) — edge added exactly where the "
            f"book is weakest.")
    if d.get("plateau_penalty") is not None:
        lines.append(
            f"Parameter sensitivity: jittering the window constants ±10% moved "
            f"the VAL IC to {[_fmt(x) for x in d.get('jitter_ics', [])]} — "
            f"plateau penalty {_fmt(d.get('plateau_penalty'))}.")
    if d.get("sign_consistency") is not None:
        lines.append(
            f"Hypothesis sign check: declared direction "
            f"{'CONFIRMED' if d['sign_consistency'] else 'CONTRADICTED'} by the "
            f"realized IC sign.")
    if d.get("complexity") is not None:
        lines.append(f"Complexity (operators + constants): {d['complexity']}.")
    defl = d.get("deflation") or {}
    if defl.get("deflated_t") is not None:
        lines.append(
            f"Multiple-testing deflation: after {d.get('n_trials')} trials the "
            f"deflated IC t-stat is {_fmt(defl.get('deflated_t'), 2)} "
            f"(luck haircut on |IC|: {_fmt(defl.get('luck_ic'))}).")

    # ── targeted advice (rule-based, from the numbers only) ──
    advice: list[str] = []
    max_corr = d.get("max_abs_corr")
    if max_corr is not None and max_corr > 0.6:
        advice.append(
            f"It is {max_corr:.0%}-correlated with an existing book factor — a "
            f"variation, not a new idea.  Change the MECHANISM (different fields, "
            f"different conditioning), not the constants.")
    mv = obj.marginal_value
    st = d.get("standalone_ic")
    if mv is not None and st is not None and mv > 0 and abs(st) < 0.01:
        advice.append(
            "Low standalone IC but positive marginal value: it works as a "
            "conditioning/state variable in combination.  Lean into that — "
            "sharpen what it conditions on rather than chasing standalone IC.")
    regime = obj.regime_independence
    if regime is not None and regime > 0 and (mv is None or regime > 2 * abs(mv or 0)):
        advice.append(
            "This is a regime specialist: it adds most of its edge in the "
            f"{d.get('regime_kind', 'drawdown')} stress bars, where the rest of "
            "the book is weak.  That is valuable diversification — preserve the "
            "crash mechanism; do not water it down to lift the calm-period IC.")
    deg = d.get("degradation_ratio")
    if deg is not None and deg < 0.5:
        advice.append(
            "The in-sample edge collapsed out-of-sample — classic overfit "
            "signature.  Simplify: fewer branches, fewer constants, a coarser "
            "window, or a more universal mechanism.")
    if d.get("plateau_penalty") is not None and d["plateau_penalty"] > 0.01:
        advice.append(
            "The edge is knife-edge in its window constants (it dies under a "
            "±10% jitter).  Real effects sit on plateaus — prefer round, "
            "economically-motivated windows and test a slower variant.")
    if d.get("sign_consistency") is False:
        advice.append(
            "The realized direction CONTRADICTS the declared hypothesis — either "
            "the mechanism is wrong (rethink it) or this is data-mining noise.  "
            "Do not just flip the sign; explain why the true direction holds.")
    ic_decay = d.get("ic_decay") or {}
    finite = {h: ic for h, ic in ic_decay.items() if ic is not None}
    if finite and st is not None:
        best_h = max(finite, key=lambda h: abs(finite[h]))
        if abs(finite[best_h]) > 1.5 * abs(st):
            advice.append(
                f"The edge is materially stronger at a {best_h}-bar horizon than "
                f"at the declared one — consider re-anchoring prediction_horizon "
                f"to {best_h}.")
    cov = d.get("coverage")
    if cov is not None and cov < 0.7:
        advice.append(
            f"Signal coverage is only {cov:.0%} of (date,ticker) cells — long "
            f"warm-ups or aggressive NaN masking.  Shorten windows or fill "
            f"defensively so the factor prices more of the panel.")

    if advice:
        lines.append("")
        lines.append("WHAT TO DO DIFFERENTLY:")
        lines.extend(f"- {a}" for a in advice)

    return "\n".join(lines)
