"""The CORE Pareto objective vector, the hard gates, and dominance primitives.

Per ``docs/research-evolution/DESIGN.md`` the selection fitness is a **Pareto front**
over four axes (no arbitrary scalar weights, which can themselves overfit), behind a
set of **hard gates** a candidate must clear before it competes at all:

CORE objective vector (all *maximised*):
  1. ``marginal_value``     — ΔOOS-IC the candidate adds to the combined model (LOCO
     against the Pareto archive).  The primary axis.
  2. ``independence``       — the candidate's **residual (orthogonalised) IC**: its
     predictive edge in the direction the book does *not* span (novel predictive
     content).  Rewards independence that actually predicts, not orthogonal noise.
     The legacy Δ-participation-ratio − soft max-|corr| penalty stays available as
     a diagnostic / ``EvalParams.independence_metric`` option.
  3. ``robustness``         — fold-refit CPCV combined/LOCO IC
     ``mean − λ·std − plateau + sign_bonus``.
  4. ``parsimony``          — ``−complexity`` (operator + constant count from the AST).
  5. ``structural_novelty`` — minimum **canonical AST weighted-subtree distance**
     (``research_eval.ast_novelty``) to the nearest archive member.  0 = structural
     clone (same canonical computation) of something already kept, 1 = no shared
     canonical subtree.  Falls back to the zoo reference distance when the archive is
     empty.  Inspired by AlphaAgent's common-subtree originality criterion but computed
     as a normalised weighted multiset overlap across all canonical subtrees (invariant
     to formatting, comments, ids, local variable names and numeric window constants;
     not exact tree-edit distance).  Promotes structural originality from a diagnostic
     to a first-class selection axis: a near-clone of an archive member is non-dominated
     only if it is strictly better on every quality axis.

Hard gates (all must pass, else the candidate is treated as dominated):
  coverage ≥ τ; OOS/IS degradation ≥ τ with matching sign; deflated-IC t-stat > 0
  given the current ``N_trials``; (optional, costed) net-of-cost IC > 0.

This module holds the *definitions* (the dataclasses, the gate booleans, the
dominance test and the non-dominated front).  The full NSGA-II ranking + crowding
distance + island bookkeeping live in the evolution controller (P1).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Sequence


# ── the CORE objective vector ────────────────────────────────────────────────

@dataclass
class ObjectiveVector:
    """The five Pareto axes for one candidate — every axis *maximised*.

    ``None`` on an axis means "not measured / not applicable"; it is treated as the
    worst possible value in dominance comparisons so an unmeasured candidate never
    dominates a measured one.
    """

    marginal_value: float | None = None
    independence: float | None = None
    robustness: float | None = None
    parsimony: float | None = None
    structural_novelty: float | None = None

    AXES = ("marginal_value", "independence", "robustness", "parsimony",
            "structural_novelty")

    def as_tuple(self) -> tuple[float, ...]:
        """The axes as floats (``None`` → ``-inf``, the worst value)."""
        return tuple(
            float("-inf") if getattr(self, a) is None else float(getattr(self, a))
            for a in self.AXES
        )  # type: ignore[return-value]

    def to_dict(self) -> dict[str, float | None]:
        return {a: getattr(self, a) for a in self.AXES}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ObjectiveVector":
        return cls(**{a: d.get(a) for a in cls.AXES})


# ── the hard gates ───────────────────────────────────────────────────────────

@dataclass
class GateResults:
    """Boolean outcomes of the hard pre-filters.

    A ``None`` gate is "not evaluated" (e.g. ``cost_ok`` when costs aren't modelled)
    and does not count against the candidate.  ``passed`` is True only when every
    *evaluated* gate is True.
    """

    coverage_ok: bool | None = None
    degradation_ok: bool | None = None
    deflation_ok: bool | None = None
    cost_ok: bool | None = None
    reasons: dict[str, str] = field(default_factory=dict)

    GATES = ("coverage_ok", "degradation_ok", "deflation_ok", "cost_ok")

    @property
    def passed(self) -> bool:
        return all(getattr(self, g) is not False for g in self.GATES)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {g: getattr(self, g) for g in self.GATES}
        d["passed"] = self.passed
        if self.reasons:
            d["reasons"] = dict(self.reasons)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GateResults":
        return cls(**{g: d.get(g) for g in cls.GATES}, reasons=d.get("reasons", {}))


@dataclass
class FitnessResult:
    """The full deterministic score for one candidate.

    ``objective`` + ``gates`` are the *selection* channel consumed by the controller;
    ``diagnostics`` is the rich *teacher* channel handed to Reflection / the LLM.
    The two are kept architecturally separate on purpose: the LLM must never see the
    numbers it is selected on in a form it can game.
    """

    candidate_id: str
    objective: ObjectiveVector
    gates: GateResults
    diagnostics: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    # QD behavior descriptors (WS2): the candidate's *behavioral* coordinates used to
    # place it in the MAP-Elites grid.  DELIBERATELY separate from ``objective`` — these
    # organise the archive for diversity, they are NOT scored on.  Keys:
    # ``trend_reversal`` (fade↔momentum), ``signal_speed`` (slow↔fast),
    # ``stress_activation`` (metadata at grid_dims=2, a 3rd bin axis at grid_dims=3).
    behavior: dict[str, Any] = field(default_factory=dict)

    @property
    def selectable(self) -> bool:
        """True if the candidate clears every hard gate (else treated as dominated)."""
        return self.gates.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "objective": self.objective.to_dict(),
            "gates": self.gates.to_dict(),
            "selectable": self.selectable,
            "diagnostics": self.diagnostics,
            "behavior": self.behavior,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FitnessResult":
        """Rebuild from :meth:`to_dict` output (the MCP / state-file round-trip)."""
        return cls(
            candidate_id=d["candidate_id"],
            objective=ObjectiveVector.from_dict(d.get("objective", {})),
            gates=GateResults.from_dict(d.get("gates", {})),
            diagnostics=d.get("diagnostics", {}),
            behavior=d.get("behavior", {}),
            raw=d.get("raw", {}),
        )


# ── Pareto dominance ─────────────────────────────────────────────────────────

def dominates(a: ObjectiveVector, b: ObjectiveVector) -> bool:
    """True if ``a`` Pareto-dominates ``b`` (maximisation on all four axes).

    ``a`` dominates ``b`` when it is ≥ on every axis and strictly > on at least one.
    """
    at, bt = a.as_tuple(), b.as_tuple()
    return all(x >= y for x, y in zip(at, bt)) and any(x > y for x, y in zip(at, bt))


def non_dominated_front(results: Sequence[FitnessResult]) -> list[FitnessResult]:
    """The Pareto front of a candidate set, honouring the hard gates.

    Candidates that fail a gate are treated as dominated, so the front is drawn only
    from gate-passing candidates whenever any exist.  If *none* pass (early in a run
    the whole population may fail), the front falls back to the non-dominated set of
    the gate-failing candidates so the search still has parents to mutate.
    """
    if not results:
        return []
    passing = [r for r in results if r.selectable]
    pool = passing if passing else list(results)

    front: list[FitnessResult] = []
    for r in pool:
        if not any(dominates(o.objective, r.objective) for o in pool if o is not r):
            front.append(r)
    return front


# ── shared numeric helpers used by the axes ──────────────────────────────────

def participation_ratio(corr: Any) -> float | None:
    """Effective number of independent factors = ``(Σλ)² / Σλ²`` of a corr matrix.

    ``n`` for a perfectly orthogonal set, ``→1`` when every factor points the same
    way.  Mirrors ``comparison/analytics.evaluate_prerun_diversity`` so the
    independence axis and the analytics track can't drift.  ``None`` for a matrix
    smaller than 1×1 or an all-zero spectrum.
    """
    import numpy as np

    C = np.asarray(corr, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1] or C.shape[0] < 1:
        return None
    C = C.copy()
    np.fill_diagonal(C, 1.0)
    C = np.nan_to_num(C, nan=0.0)
    C = (C + C.T) / 2.0
    eig = np.clip(np.linalg.eigvalsh(C), 0.0, None)
    denom = float((eig ** 2).sum())
    if denom <= 0:
        return None
    return min(float(eig.sum() ** 2 / denom), float(C.shape[0]))


def complexity(code: str) -> int:
    """Parse a factor's Python source and count operators + free constants.

    Parsimony proxy for the fourth Pareto axis (reward ``−complexity``): overfit
    factors accrete operators and magic constants, economically-motivated ones stay
    lean.  Counts binary/unary/boolean/compare operators, function calls and numeric
    constants across the whole module AST.  Unparseable code returns a large sentinel
    so it is penalised, never crashes the harness.
    """
    if not code or not code.strip():
        return 0
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 10_000
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Call)):
            n += 1
        elif isinstance(node, ast.BoolOp):
            n += max(1, len(node.values) - 1)
        elif isinstance(node, ast.Compare):
            n += len(node.ops)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            n += 1
    return n
