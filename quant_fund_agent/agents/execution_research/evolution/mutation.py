"""Deterministic mutation operators for execution programs (E1: param jitter).

The ``params = {...}`` class-attribute dict is the declared **jitter surface**
of every executor (DESIGN §Genome) — the execution twin of the factor arm's
window-jitter.  The same AST operator doubles as:

* the **jitter mutation** (`random_jitter_child`): each numeric param scaled by
  an independent ``U(1−pct, 1+pct)`` draw, and
* the **plateau probe** (`jitter_variants`): fixed ±scale variants whose VAL
  fitness the harness reports as a diagnostic — a program whose Sharpe
  collapses under a ±10% param nudge sits on a knife-edge, not a plateau.

``rewrite_factor_id`` is reused from the factor arm's mutation module — it
replaces string constants equal to the old id, which is exactly what renaming
an ``executor_id`` needs.
"""

from __future__ import annotations

import ast
import copy
from typing import Sequence

import numpy as np

from quant_fund_agent.agents.execution_research.evolution.genome import (
    ExecutionProgram,
)
from quant_fund_agent.agents.factor_research.evolution.mutation import (
    rewrite_factor_id,
)


def _params_dict_node(tree: ast.Module) -> ast.Dict | None:
    """Locate the ``params = {...}`` class-attr dict of the executor class."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {getattr(b, "id", getattr(b, "attr", "")) for b in node.bases}
        if "BaseExecutor" not in bases:
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "params" \
                            and isinstance(stmt.value, ast.Dict):
                        return stmt.value
    return None


def param_constants(code: str) -> dict[str, float]:
    """The numeric params an executor declares — its jitter surface."""
    node = _params_dict_node(ast.parse(code))
    if node is None:
        return {}
    out: dict[str, float] = {}
    for k, v in zip(node.keys, node.values):
        if isinstance(k, ast.Constant) and isinstance(k.value, str) \
                and isinstance(v, ast.Constant) \
                and isinstance(v.value, (int, float)) \
                and not isinstance(v.value, bool):
            out[k.value] = float(v.value)
    return out


def jitter_params(code: str, scales: Sequence[float] | float) -> tuple[str, int]:
    """Scale the numeric values of the ``params`` dict literal → (code, n_changed).

    ``scales`` is either one factor for every param or a per-param sequence
    (ordered as the dict literal).  Ints stay ints (rounded, floored at 1 so a
    holding period can never hit 0); floats scale freely.  ``n_changed`` counts
    params whose value actually moved — 0 means there is no jitterable surface
    and the caller should skip rather than treat the identical program as a
    variant.
    """
    tree = ast.parse(code)
    node = _params_dict_node(tree)
    if node is None:
        return code, 0

    if isinstance(scales, (int, float)):
        scale_list = [float(scales)] * len(node.values)
    else:
        scale_list = [float(s) for s in scales]

    n_changed = 0
    idx = 0
    for k, v in zip(node.keys, node.values):
        if not (isinstance(k, ast.Constant) and isinstance(v, ast.Constant)
                and isinstance(v.value, (int, float))
                and not isinstance(v.value, bool)):
            continue
        s = scale_list[idx % len(scale_list)]
        idx += 1
        old = v.value
        if isinstance(old, int):
            new = max(1, int(round(old * s)))
        else:
            new = float(old * s)
        if new != old:
            v.value = new
            n_changed += 1
    return ast.unparse(ast.fix_missing_locations(tree)), n_changed


def jitter_variants(program: ExecutionProgram,
                    scales: Sequence[float] = (0.9, 1.1),
                    ) -> list[tuple[str, str]]:
    """Plateau-probe variants → ``[(variant_id, code), ...]`` (or ``[]``).

    One variant per scale, ``executor_id`` rewritten to a unique probe id so
    in-memory compilation never collides with the candidate itself.
    """
    out: list[tuple[str, str]] = []
    for k, scale in enumerate(scales):
        new_code, n = jitter_params(program.code, scale)
        if n == 0:
            return []
        probe_id = f"{program.executor_id}_jit{k}"
        out.append((probe_id,
                    rewrite_factor_id(new_code, program.executor_id, probe_id)))
    return out


def random_jitter_child(program: ExecutionProgram, rng: np.random.Generator,
                        new_id: str, pct: float = 0.15,
                        ) -> ExecutionProgram | None:
    """The jitter *mutation*: every param scaled by an independent U(1−pct, 1+pct).

    Returns a child (same mechanism, nudged params, new id) or ``None`` when
    the parent declares no jitterable params.
    """
    n_params = len(param_constants(program.code))
    if n_params == 0:
        return None
    scales = [float(rng.uniform(1.0 - pct, 1.0 + pct)) for _ in range(n_params)]
    new_code, n = jitter_params(program.code, scales)
    if n == 0:
        return None
    child = copy.deepcopy(program)
    child.executor_id = new_id
    child.code = rewrite_factor_id(new_code, program.executor_id, new_id)
    child.name = f"{program.name} (jitter)" if program.name else new_id
    return child


# ── E2: LLM-semantic mutation prompts (the creative operator) ─────────────────

EXEC_CONTRACT = '''\
THE EXECUTOR CONTRACT (follow it exactly):
- Emit ONE complete Python module defining ONE class decorated with
  @register_executor, subclassing BaseExecutor, imported as:
      from quant_fund_agent.execution.base import BaseExecutor, register_executor
- Class attributes: executor_id (snake_case, must equal the id you are given),
  name, regime ("cross_sectional" builds a dollar-neutral long/short book;
  "per_underlying" a directional per-name book), inputs (state fields you read),
  params = {...} — a dict of NUMERIC constants only. Put EVERY tunable number in
  params and read them via `p = type(self).params` — params is the mutation
  surface for the search.
- Implement `target_weights(self, signal, state)` for path-INdependent logic:
  (T x N) DataFrame signal + state dict -> (T x N) DataFrame of target weights.
  For path-DEPENDENT logic (stops, profit-taking, position-aware rebalancing)
  implement `step(self, t, signal_row, state_row, book)` -> Series instead;
  `book.positions`, `book.unrealised_pnl`, `book.drawdown` carry your own book.
- `state` / `state_row` fields (ALL causal, computed for you): "vol" (trailing
  volatility), "adv" (liquidity), "drawdown" (per-name price drawdown, <= 0),
  "signal_age" (bars since the signal changed); "spread" when the feed has it.
- CAUSALITY IS A HARD GATE: weights at bar t are re-checked on a truncated
  panel. NO full-sample mean/std, no centred windows, nothing that reads rows
  after t. Use expanding/rolling(trailing) statistics only.
- Output constraints (hard gates): finite weights, |w| <= 1 per name, gross
  sum |w| <= 2, and for cross_sectional regime the book must be ~dollar-neutral.
  Excessive turnover is penalised through transaction costs - hysteresis
  (separate entry/exit bands) and slower rebalancing are your friends.
- Allowed imports: numpy, pandas, scipy, sklearn, statsmodels,
  quant_fund_agent.execution.base. Nothing else (no os/open/requests/...).
'''


def _known_ids_note(known_ids) -> str:
    if not known_ids:
        return ""
    sample = ", ".join(sorted(known_ids)[:40])
    return f"\nIDs already taken (do NOT reuse): {sample}\n"


def build_exec_mutation_prompt(parent, brief: str, new_id: str,
                               known_ids=()) -> str:
    """One parent + its deterministic brief → ask for a mutated child program."""
    return f'''You are the Execution Researcher of a quant fund: you evolve the
PROGRAM that turns a strategy's composite alpha signal into a target book
through time (position sizing, entry/exit bands, risk overlays) — you never
touch the alpha itself.

PARENT PROGRAM (id: {parent.executor_id}, regime: {parent.regime}):
mechanism: {parent.mechanism or "(none declared)"}
```python
{parent.code}
```

DETERMINISTIC EVALUATION BRIEF OF THE PARENT:
{brief or "(first evaluation - no brief yet)"}

{EXEC_CONTRACT}
Mutate the parent into ONE improved child program. Address the brief's advice
with a REAL mechanism change (state-conditional sizing, hysteresis, decay-aware
holding, drawdown de-risking, turnover budgeting...), not a constant tweak —
the search has a separate operator for constants.{_known_ids_note(known_ids)}
Use executor_id "{new_id}" exactly.

Respond with ONLY a JSON object (no markdown fences):
{{"executor_id": "{new_id}", "name": "...", "regime": "cross_sectional|per_underlying",
 "mechanism": "the execution idea in one sentence",
 "expected_effect": "a falsifiable claim, e.g. 'cuts turnover >=20% at <=10% capture loss'",
 "code": "the complete Python module as one string"}}'''


def build_exec_crossover_prompt(a, brief_a: str, b, brief_b: str, new_id: str,
                                known_ids=()) -> str:
    """Two parents → ask for a child combining their execution mechanisms."""
    return f'''You are the Execution Researcher of a quant fund: you evolve
signal→book execution programs (see contract below).

PARENT A (id: {a.executor_id}, regime: {a.regime}, mechanism: {a.mechanism or "?"}):
```python
{a.code}
```
Brief A:
{brief_a or "(none)"}

PARENT B (id: {b.executor_id}, regime: {b.regime}, mechanism: {b.mechanism or "?"}):
```python
{b.code}
```
Brief B:
{brief_b or "(none)"}

{EXEC_CONTRACT}
Combine the two parents' MECHANISMS into ONE child program that plausibly keeps
the strengths of both (e.g. A's entry logic with B's risk overlay). Do not
concatenate code blindly.{_known_ids_note(known_ids)}
Use executor_id "{new_id}" exactly.

Respond with ONLY a JSON object (no markdown fences):
{{"executor_id": "{new_id}", "name": "...", "regime": "cross_sectional|per_underlying",
 "mechanism": "...", "expected_effect": "...",
 "code": "the complete Python module as one string"}}'''


def parse_exec_child_response(text: str) -> dict:
    """Parse the LLM's JSON child response (tolerates markdown fences).

    Raises ``ValueError`` on anything unusable — the caller logs and skips
    (a bad generation is a data point, never a crash).
    """
    import json as _json
    import re as _re

    raw = text.strip()
    fence = _re.search(r"```(?:json)?\s*(.*?)```", raw, _re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in response")
    payload = _json.loads(raw[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("response JSON is not an object")
    for key in ("executor_id", "code"):
        if not payload.get(key) or not isinstance(payload[key], str):
            raise ValueError(f"response JSON lacks a usable {key!r}")
    if payload.get("regime") not in ("cross_sectional", "per_underlying"):
        payload["regime"] = "per_underlying"
    return payload
