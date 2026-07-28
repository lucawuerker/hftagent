"""Mutation operators: programmatic window-jitter + LLM-semantic mutate/crossover.

Two kinds of operator, per the design:

* **Window-jitter** (:func:`jitter_windows`, :func:`jitter_variants`) — a cheap,
  fully deterministic AST transform that scales the integer *window* constants a
  factor passes to the time-series ops (``ts_mean(x, 20)`` → ``ts_mean(x, 22)``).
  It plays two roles: a free mutation operator, and the probe for the
  parameter-sensitivity **plateau penalty** (evaluate the ±10% variants' VAL IC;
  a knife-edge factor collapses, a plateau factor doesn't).
* **LLM-semantic** (:func:`build_mutation_prompt`, :func:`parse_child_response`)
  — the creative operator: the LLM is shown one parent (mutation) or two
  (crossover) *with their deterministic reflection briefs* (the teacher channel)
  and asked for an improved child program in one call, FunSearch-style.  The LLM
  proposes; it never scores — the child goes straight to the deterministic
  harness.

The AST transforms use ``ast.unparse``, which drops ``#`` comments (docstrings
survive).  Jitter probes are in-memory only so this is invisible; a jitter child
that survives to persistence is stored in its unparsed form, which is still
valid, readable Python.
"""

from __future__ import annotations

import ast
import copy
import logging
from typing import Any, Sequence

import numpy as np

from quant_fund_agent.agents.factor_research.evolution.genome import FactorProgram

log = logging.getLogger("evolution.mutation")

# ops-function → index of its positional window argument (from factors/ops.py).
_WINDOW_ARG: dict[str, int] = {
    "delta": 1, "delay": 1, "ts_sum": 1, "ts_mean": 1, "stddev": 1,
    "ts_min": 1, "ts_max": 1, "ts_argmax": 1, "ts_argmin": 1, "ts_rank": 1,
    "product": 1, "decay_linear": 1, "adv": 1,
    "correlation": 2, "covariance": 2,
}
# DataFrame methods whose first positional int arg is a bar count.
_WINDOW_METHODS = {"rolling", "shift", "diff"}


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


class _WindowScaler(ast.NodeTransformer):
    """Scale every window-int in ops calls / rolling-style methods.

    ``scales`` yields one multiplicative factor per window encountered (in AST
    walk order); a scaled window is clamped to ≥ 2 and nudged by ±1 when
    rounding lands back on the original value, so every "jitter" genuinely
    changes the program.
    """

    def __init__(self, scales: "list[float]") -> None:
        self._scales = scales
        self._i = 0
        self.n_changed = 0

    def _next_scale(self) -> float:
        s = self._scales[self._i % len(self._scales)]
        self._i += 1
        return s

    def _scaled(self, value: int, scale: float) -> int:
        new = max(2, int(round(value * scale)))
        if new == value:
            new = max(2, value + (1 if scale >= 1.0 else -1))
        return new

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        name = _call_name(node)
        arg_idx: int | None = None
        if isinstance(node.func, ast.Name) and name in _WINDOW_ARG:
            arg_idx = _WINDOW_ARG[name]
        elif isinstance(node.func, ast.Attribute) and name in _WINDOW_METHODS:
            arg_idx = 0
        if arg_idx is None or arg_idx >= len(node.args):
            return node
        arg = node.args[arg_idx]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, int) \
                and not isinstance(arg.value, bool) and arg.value >= 2:
            new_val = self._scaled(arg.value, self._next_scale())
            if new_val != arg.value:
                node.args[arg_idx] = ast.copy_location(ast.Constant(new_val), arg)
                self.n_changed += 1
        return node


def window_constants(code: str) -> list[int]:
    """The integer window constants a factor passes to the window-taking ops."""
    out: list[int] = []
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        arg_idx: int | None = None
        if isinstance(node.func, ast.Name) and name in _WINDOW_ARG:
            arg_idx = _WINDOW_ARG[name]
        elif isinstance(node.func, ast.Attribute) and name in _WINDOW_METHODS:
            arg_idx = 0
        if arg_idx is None or arg_idx >= len(node.args):
            continue
        arg = node.args[arg_idx]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, int) \
                and not isinstance(arg.value, bool) and arg.value >= 2:
            out.append(arg.value)
    return out


def rewrite_factor_id(code: str, old_id: str, new_id: str) -> str:
    """Return ``code`` with the ``factor_id`` class attribute set to ``new_id``.

    Only the exact string constant assigned to ``factor_id`` (and any other
    string constant equal to ``old_id``, e.g. in the docstring header) is
    replaced — an AST transform, not a text substitution, so identifiers that
    merely contain the old id are untouched.
    """
    tree = ast.parse(code)

    class _Rewriter(ast.NodeTransformer):
        def visit_Constant(self, node: ast.Constant):  # noqa: N802
            if isinstance(node.value, str) and node.value == old_id:
                return ast.copy_location(ast.Constant(new_id), node)
            return node

    new_tree = ast.fix_missing_locations(_Rewriter().visit(tree))
    return ast.unparse(new_tree)


def jitter_windows(code: str, scale: float) -> tuple[str, int]:
    """Deterministically scale every window constant by ``scale``.

    Returns ``(new_code, n_windows_changed)``.  ``n_windows_changed == 0`` means
    the factor has no jitterable windows (e.g. purely cross-sectional) — the
    caller should then skip the plateau probe rather than treat the identical
    program as a variant.
    """
    tree = ast.parse(code)
    scaler = _WindowScaler([scale])
    new_tree = ast.fix_missing_locations(scaler.visit(tree))
    return ast.unparse(new_tree), scaler.n_changed


def jitter_variants(program: FactorProgram, scales: Sequence[float] = (0.9, 1.1),
                    ) -> list[tuple[str, str]]:
    """The plateau-probe variants of a program → ``[(variant_id, code), ...]``.

    One variant per scale, with the ``factor_id`` rewritten to a unique probe id
    so in-memory compilation never collides with the candidate itself.  Programs
    with no windows return ``[]``.
    """
    out: list[tuple[str, str]] = []
    for k, scale in enumerate(scales):
        new_code, n = jitter_windows(program.code, scale)
        if n == 0:
            return []
        probe_id = f"{program.factor_id}_jit{k}"
        out.append((probe_id, rewrite_factor_id(new_code, program.factor_id, probe_id)))
    return out


def random_jitter_child(program: FactorProgram, rng: np.random.Generator,
                        new_id: str, pct: float = 0.10) -> FactorProgram | None:
    """The jitter *mutation* operator: each window scaled by U(1−pct, 1+pct).

    Returns a child :class:`FactorProgram` (same hypothesis, new windows, new
    id) or ``None`` when the parent has no jitterable windows.
    """
    n_windows = len(window_constants(program.code))
    if n_windows == 0:
        return None
    scales = [float(rng.uniform(1.0 - pct, 1.0 + pct)) for _ in range(n_windows)]
    tree = ast.parse(program.code)
    scaler = _WindowScaler(scales)
    new_tree = ast.fix_missing_locations(scaler.visit(tree))
    code = rewrite_factor_id(ast.unparse(new_tree), program.factor_id, new_id)
    child = copy.deepcopy(program)
    child.factor_id = new_id
    child.code = code
    child.name = f"{program.name} (jitter)" if program.name else new_id
    return child


# ── creative mode (--creative-frac): a clearly-marked extra prompt section ────

CREATIVE_NOVELTY_SECTION = """\
CREATIVE MODE — PROPOSE A GENUINELY NOVEL MECHANISM
---------------------------------------------------
For THIS child, do NOT derive the mechanism from the parent factor(s) or from
any paper referenced above.  Propose a genuinely NOVEL mechanism of your own,
drawing on mathematics, statistics, signal processing, physics or behavioural
reasoning (for example: Hawkes-process self-excitation of order flow, spectral
coherence between volume and returns, Ornstein-Uhlenbeck relaxation rates,
distributional entropy shifts, prospect-theory anchoring) — explicitly NOT just
another trend-following / moving-average variant.  The mechanism must remain
causal, leak-free and computable from the fields in the DATA CONTEXT, and you
must respond with the exact same JSON schema requested in this prompt."""


# ── hypothesis-only mutation (P3 agent split: Hypothesis → Debate → Codegen) ──

HYPOTHESIS_MUTATION_PROMPT = """\
You are a senior quantitative researcher running an evolutionary search over
alpha-factor programs.  Below is a PARENT factor and the deterministic
evaluation report our research harness produced for it.  Your task is to
propose the HYPOTHESIS for one improved CHILD factor — the mechanism, not the
code (implementation happens in a later step, and a skeptic will attack your
hypothesis first, so make it defensible).

Keep what works, fix what the report criticises.  If the report shows the
parent is redundant with the book, change the *mechanism*, not the constants.
The mechanism may be 2-4 sentences and may be economic OR mathematical /
statistical / microstructural / behavioural — reasoning like "Hawkes-process
self-excitation describes clustering of order flow" is as valid as a classic
economic story.  State a falsifiable directional claim (`expected_sign`) and
when the effect should work or fail (`regime_dependence`).

DATA CONTEXT
------------
{data_context}

PARENT FACTOR
-------------
factor_id: {parent_id}
hypothesis: {parent_idea}
expected_sign: {parent_sign}
prediction_horizon: {parent_horizon}

```python
{parent_code}
```

EVALUATION REPORT (deterministic, out-of-sample)
------------------------------------------------
{brief}

EXISTING FACTOR IDS (do NOT reuse)
----------------------------------
{existing_ids}

OUTPUT FORMAT
-------------
Respond with strict JSON, exactly:
{idea_schema}
"""


def build_hypothesis_prompt(parent: FactorProgram, brief: str, data_context: str,
                            existing_ids: Sequence[str],
                            creative: bool = False) -> str:
    from quant_fund_agent.agents.factor_research.debate import IDEA_SCHEMA

    prompt = HYPOTHESIS_MUTATION_PROMPT.format(
        data_context=data_context,
        parent_id=parent.factor_id,
        parent_idea=parent.trading_idea or parent.description or "(none recorded)",
        parent_sign=parent.expected_sign if parent.expected_sign is not None else "(none)",
        parent_horizon=parent.prediction_horizon,
        parent_code=parent.code,
        brief=brief or "(first evaluation pending)",
        existing_ids=", ".join(sorted(existing_ids)) if existing_ids else "(none)",
        idea_schema=IDEA_SCHEMA,
    )
    if creative:
        prompt = f"{prompt}\n\n{CREATIVE_NOVELTY_SECTION}"
    return prompt


def parse_hypothesis_response(content: str) -> dict[str, Any]:
    """Parse the hypothesis JSON into a raw idea dict (raises on junk)."""
    from quant_fund_agent.agents.factor_research.graph import _parse_json

    raw = _parse_json(content)
    fid = (raw.get("factor_id") or "").strip().lower()
    if not fid or not (raw.get("trading_idea") or "").strip():
        raise ValueError("hypothesis response missing factor_id or trading_idea")
    raw["factor_id"] = fid
    return raw


# ── LLM-semantic mutation / crossover ─────────────────────────────────────────

CHILD_SCHEMA = """\
OUTPUT FORMAT
-------------
Respond with strict JSON, exactly:
{{
  "factor_id": "<new unique snake_case id, NOT any existing id>",
  "name": "<short human-readable name>",
  "category": "<momentum|mean_reversion|volatility|microstructure|statistical_arbitrage|carry|sentiment|other>",
  "trading_idea": "<2-4 sentences: WHY the child should have edge — the mechanism>",
  "description": "<1-2 sentences: what the signal computes>",
  "prediction_horizon": <positive int: bars ahead the edge peaks>,
  "suggested_horizons": [<int>, ...],
  "expected_sign": <1 or -1: the direction you predict — +1 means higher signal
                    values should precede higher forward returns>,
  "code": "<the COMPLETE .py file contents for the child factor, one string>"
}}

The "code" string must follow every rule in the STRICT REQUIREMENTS section.
Do NOT wrap it in markdown fences."""


MUTATION_PROMPT = """\
You are a senior quantitative researcher running an evolutionary search over
alpha-factor programs.  Below is a PARENT factor and the deterministic
evaluation report our research harness produced for it.  Your task is to
propose ONE improved CHILD factor: keep what works, fix what the report
criticises, and state a falsifiable hypothesis for the child.

Think like a scientist, not a curve-fitter: the child must embody a clearly
stated mechanism (2-4 sentences), not just perturbed constants.  The mechanism
may be economic OR mathematical / statistical / microstructural / behavioural —
reasoning like "Hawkes-process self-excitation describes clustering of order
flow" is as valid as a classic economic story.  If the report shows the parent is
redundant with the book, change the *mechanism*, not the window.  If the edge
decays instantly, look for a slower expression of the same idea (or declare a
shorter horizon).  If the parent failed gates badly, it is fine to propose a
substantially different factor inspired by the parent's weaknesses.

DATA CONTEXT
------------
{data_context}

{operator_reference}

{example_factor}

PARENT FACTOR
-------------
factor_id: {parent_id}
hypothesis: {parent_idea}
expected_sign: {parent_sign}
prediction_horizon: {parent_horizon}

```python
{parent_code}
```

EVALUATION REPORT (deterministic, out-of-sample)
------------------------------------------------
{brief}

EXISTING FACTOR IDS (do NOT reuse)
----------------------------------
{existing_ids}

STRICT REQUIREMENTS
-------------------
{strict_requirements}

{child_schema}
"""


CROSSOVER_PROMPT = """\
You are a senior quantitative researcher running an evolutionary search over
alpha-factor programs.  Below are TWO parent factors with their deterministic
evaluation reports.  Your task is to propose ONE child factor that *combines
the complementary strengths* of both parents — a genuine synthesis of their
mechanisms (e.g. one parent's signal conditioned on the other's state
variable), not a naive average of their formulas.

DATA CONTEXT
------------
{data_context}

{operator_reference}

{example_factor}

PARENT A
--------
factor_id: {parent_a_id}
hypothesis: {parent_a_idea}

```python
{parent_a_code}
```

EVALUATION REPORT A
-------------------
{brief_a}

PARENT B
--------
factor_id: {parent_b_id}
hypothesis: {parent_b_idea}

```python
{parent_b_code}
```

EVALUATION REPORT B
-------------------
{brief_b}

EXISTING FACTOR IDS (do NOT reuse)
----------------------------------
{existing_ids}

STRICT REQUIREMENTS
-------------------
{strict_requirements}

{child_schema}
"""

# The codegen contract, restated once for the evolution prompts (kept in sync
# with CODEGEN_PROMPT's STRICT REQUIREMENTS — same validator enforces both).
STRICT_REQUIREMENTS = """\
1. Only import from: ``pandas`` (as pd), ``numpy`` (as np),
   ``scipy`` / ``statsmodels`` for deterministic paper math,
   ``quant_fund_agent.factors.base`` (BaseFactor),
   ``quant_fund_agent.factors.registry`` (register_factor),
   ``quant_fund_agent.factors.ops`` (the helper operators), ``__future__``.
2. Define exactly ONE class subclassing ``BaseFactor``, decorated with
   ``@register_factor``, whose ``factor_id`` equals the "factor_id" you return.
3. The class MUST declare ``inputs = [...]`` covering every field read via
   ``data["..."]``, and ``prediction_horizon = <positive int>`` matching the
   "prediction_horizon" you return.
4. ``calc(self, data)`` returns a numeric DataFrame with the same index and
   columns as ``data["close"]``; guard sparse fields defensively.
5. Every ``ops`` helper is positional-only — ``ts_sum(x, 5)``, never
   ``ts_sum(x, n=5)``.  Pandas methods are called on the DataFrame directly.
6. If the paper/mechanism needs a transform not in ``ops``, implement a small
   deterministic helper in the file.  It must be causal: trailing windows,
   positive lags, no centered rolling, no negative shifts, and no full-panel
   sklearn/statsmodels ``fit`` inside ``calc``.
7. No I/O, no network, no ``os``/``open``/``eval``/``exec``/``__import__``."""


def build_mutation_prompt(parent: FactorProgram, brief: str, data_context: str,
                          existing_ids: Sequence[str],
                          creative: bool = False) -> str:
    from quant_fund_agent.agents.factor_research.prompts import (
        EXAMPLE_FACTOR,
        OPERATOR_REFERENCE,
    )

    prompt = MUTATION_PROMPT.format(
        data_context=data_context,
        operator_reference=OPERATOR_REFERENCE,
        example_factor=EXAMPLE_FACTOR,
        parent_id=parent.factor_id,
        parent_idea=parent.trading_idea or parent.description or "(none recorded)",
        parent_sign=parent.expected_sign if parent.expected_sign is not None else "(none)",
        parent_horizon=parent.prediction_horizon,
        parent_code=parent.code,
        brief=brief or "(first evaluation pending)",
        existing_ids=", ".join(sorted(existing_ids)) if existing_ids else "(none)",
        strict_requirements=STRICT_REQUIREMENTS,
        child_schema=CHILD_SCHEMA,
    )
    if creative:
        prompt = f"{prompt}\n\n{CREATIVE_NOVELTY_SECTION}"
    return prompt


def build_crossover_prompt(parent_a: FactorProgram, brief_a: str,
                           parent_b: FactorProgram, brief_b: str,
                           data_context: str, existing_ids: Sequence[str]) -> str:
    from quant_fund_agent.agents.factor_research.prompts import (
        EXAMPLE_FACTOR,
        OPERATOR_REFERENCE,
    )

    return CROSSOVER_PROMPT.format(
        data_context=data_context,
        operator_reference=OPERATOR_REFERENCE,
        example_factor=EXAMPLE_FACTOR,
        parent_a_id=parent_a.factor_id,
        parent_a_idea=parent_a.trading_idea or parent_a.description or "(none recorded)",
        parent_a_code=parent_a.code,
        brief_a=brief_a or "(no report)",
        parent_b_id=parent_b.factor_id,
        parent_b_idea=parent_b.trading_idea or parent_b.description or "(none recorded)",
        parent_b_code=parent_b.code,
        brief_b=brief_b or "(no report)",
        existing_ids=", ".join(sorted(existing_ids)) if existing_ids else "(none)",
        strict_requirements=STRICT_REQUIREMENTS,
        child_schema=CHILD_SCHEMA,
    )


def parse_child_response(content: str) -> FactorProgram:
    """Parse the LLM's child JSON into a :class:`FactorProgram` (raises on junk)."""
    from quant_fund_agent.agents.factor_research.graph import (
        _coerce_horizon,
        _coerce_horizon_list,
        _parse_json,
    )

    raw: dict[str, Any] = _parse_json(content)
    fid = (raw.get("factor_id") or "").strip().lower()
    code = raw.get("code") or ""
    if not fid or not code.strip():
        raise ValueError("child response missing factor_id or code")

    sign_raw = raw.get("expected_sign")
    try:
        sign = int(np.sign(int(sign_raw))) or None if sign_raw is not None else None
    except (TypeError, ValueError):
        sign = None

    return FactorProgram(
        factor_id=fid,
        code=code,
        name=raw.get("name", fid),
        category=raw.get("category", "other"),
        trading_idea=raw.get("trading_idea", ""),
        description=raw.get("description", ""),
        prediction_horizon=_coerce_horizon(raw.get("prediction_horizon"), 6),
        suggested_horizons=_coerce_horizon_list(raw.get("suggested_horizons")),
        expected_sign=sign,
        source_paper_ids=list(raw.get("source_paper_ids", []) or []),
    )
