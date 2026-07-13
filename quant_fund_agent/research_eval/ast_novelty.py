"""Canonical AST weighted-subtree novelty — the structural-originality metric.

This module measures the *structural* similarity between the computations two
factor programs implement, independent of irrelevant source-level differences
(formatting, comments, docstrings, factor ids, class names, imports, local
variable names, numeric window constants, harmless temporary assignments).

It replaces the previous whitespace-stripped ``difflib.SequenceMatcher`` proxy,
which measured character-sequence similarity, not structural similarity.

Approach (inspired by AlphaAgent's common-subtree originality criterion, but
generalised): we do **not** take only the single largest common subtree — that
biases toward program size.  Instead we build, for each program, a *weighted
multiset of every canonical rooted subtree* and compare two programs with a
weighted multiset Jaccard overlap.  This is **not** exact tree-edit distance
(which is polynomial and unnecessary here); it is a linear-time, cacheable,
deterministic subtree-overlap measure.

Pipeline (all deterministic):

1. **Extract** the factor computation — the body of the ``calc`` method/function
   if present, else a best-effort fallback to the module's executable statements.
   Imports, decorators, class names, metadata assignments (``factor_id`` /
   ``name`` / ``category`` / ``inputs`` / ``prediction_horizon``), docstrings and
   type annotations are never compared.
2. **Canonicalise** the computation (:func:`canonical_factor_tree`):
   * strip source-location attributes, docstrings and ``pass`` nodes;
   * alpha-rename local variables + parameters to positional placeholders
     (``close``/``volume`` data fields, operator/function/attribute names and
     ``data["close"]``-style string keys are preserved);
   * inline safe straight-line single-use temporaries into the returned
     expression (only when no control flow / re-assignment makes it ambiguous);
   * replace numeric literal *values* with typed placeholders (``INT``/``FLOAT``/
     ``COMPLEX``) so a 20-bar vs 21-bar window is a parameter mutation, not a new
     structure (booleans / ``None`` / strings / field names are preserved);
   * canonicalise the child order of the commutative ``Add`` and ``Mult``
     operators using deterministic subtree serialisations (``Sub`` / ``Div`` /
     comparisons / call arguments are never reordered).
3. **Fingerprint** every canonical subtree deterministically (SHA-256 over the
   node label + child fingerprints) into an immutable multiset profile
   (:func:`subtree_profile`), recording each fingerprint's multiplicity and
   subtree size.
4. **Compare** two profiles (:func:`ast_subtree_similarity`) with the weighted
   multiset Jaccard

       S(P,Q) = Σ_h w(h)·min(c_P(h), c_Q(h)) / Σ_h w(h)·max(c_P(h), c_Q(h)),
       w(h)   = 1 + log(1 + size(h)),

   and distance ``d = 1 − S`` (:func:`ast_subtree_distance`).

Properties: ``0 ≤ S,d ≤ 1``; symmetric; identical canonical computations →
``S=1, d=0``; programs sharing larger computational subtrees score more
similarity than those sharing only trivial leaves.  Empty / syntactically
invalid source returns ``None`` (never crashes the evaluator), so callers keep
the "axis unmeasured" semantics rather than inventing a score.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

NOVELTY_METRIC = "canonical_ast_weighted_subtree_jaccard"

# Typed placeholders that replace numeric literal *values* during
# canonicalisation.  Tuples can never appear as a ``Constant.value`` produced by
# ``ast.parse`` (source constants are int/float/complex/str/bytes/bool/None), so
# they are unambiguous sentinels.
_INT_PLACEHOLDER = ("__num__", "int")
_FLOAT_PLACEHOLDER = ("__num__", "float")
_COMPLEX_PLACEHOLDER = ("__num__", "complex")

# AST marker singletons captured in a node's *label* rather than recursed into as
# children (operators, comparison ops, load/store contexts).
_LABEL_ONLY = (ast.expr_context, ast.operator, ast.unaryop, ast.boolop, ast.cmpop)


# ── 1. extract the factor computation ────────────────────────────────────────

def _find_calc(tree: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """First function/method named ``calc`` anywhere in the module (deterministic)."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "calc":
            return node
    return None


def _param_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """All parameter names of ``fn`` in signature order (``self``/``data``/…)."""
    a = fn.args
    names: list[str] = []
    for arg in [*getattr(a, "posonlyargs", []), *a.args]:
        names.append(arg.arg)
    if a.vararg:
        names.append(a.vararg.arg)
    for arg in a.kwonlyargs:
        names.append(arg.arg)
    if a.kwarg:
        names.append(a.kwarg.arg)
    return names


def _extract(code: str) -> tuple[list[str], list[ast.stmt]] | None:
    """Parse ``code`` → ``(param_names, body_statements)`` of the factor computation.

    Prefers the ``calc`` method's body; falls back to the module's executable
    statements (imports / class + function defs excluded).  Returns ``None`` for
    empty or syntactically invalid source.
    """
    if not code or not code.strip():
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    calc = _find_calc(tree)
    if calc is not None:
        return _param_names(calc), list(calc.body)
    # best-effort fallback: module-level executable statements only.
    stmts = [
        s for s in tree.body
        if not isinstance(s, (ast.Import, ast.ImportFrom, ast.ClassDef,
                              ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return [], stmts


def _strip_leading_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        return body[1:]
    return body


# ── 2. canonicalise ──────────────────────────────────────────────────────────

def _pre_order_names(params: list[str], body: list[ast.stmt]) -> list[str]:
    """Bound-name identifiers in first-appearance order (params first)."""
    bound: set[str] = set()
    for node in body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                bound.add(sub.id)
            elif isinstance(sub, ast.arg):  # nested comprehensions never reach here
                bound.add(sub.arg)
    order: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        if name in bound and name not in seen:
            seen.add(name)
            order.append(name)

    for p in params:  # parameters get the first canonical slots, in signature order
        bound.add(p)
        _add(p)

    def _visit(node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            _add(node.id)
        for child in ast.iter_child_nodes(node):
            _visit(child)

    for stmt in body:
        _visit(stmt)
    return order


class _Canonicaliser(ast.NodeTransformer):
    """Alpha-rename bound names, placeholder numeric literals, drop annotations."""

    def __init__(self, rename: dict[str, str]):
        self.rename = rename

    def visit_Name(self, node: ast.Name) -> ast.AST:
        new_id = self.rename.get(node.id, node.id)
        return ast.copy_location(ast.Name(id=new_id, ctx=node.ctx), node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        v = node.value
        if isinstance(v, bool):
            return node  # preserve booleans
        if isinstance(v, int):
            placeholder = _INT_PLACEHOLDER
        elif isinstance(v, float):
            placeholder = _FLOAT_PLACEHOLDER
        elif isinstance(v, complex):
            placeholder = _COMPLEX_PLACEHOLDER
        else:
            return node  # preserve str / bytes / None
        return ast.copy_location(ast.Constant(value=placeholder), node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        # drop the annotation; keep the assignment if it has a value
        node = self.generic_visit(node)  # type: ignore[assignment]
        if node.value is None:
            return None  # a bare ``x: T`` declaration carries no computation
        return ast.copy_location(
            ast.Assign(targets=[node.target], value=node.value), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.annotation = None
        node.type_comment = None
        return node


def _inline_straightline(body: list[ast.stmt]) -> ast.expr | None:
    """Inline safe single-assignment temporaries into one result expression.

    Only fires when the whole body is straight-line: a run of single-``Name``-
    target assignments ending in a result (``return`` value / trailing ``Expr`` /
    trailing ``Assign``), with every assigned name assigned exactly once and no
    control flow, augmented assignment, tuple unpacking, deletion, etc.  Returns
    the fully-substituted result expression, or ``None`` when inlining is unsafe
    (the caller then keeps the canonicalised statement tree).
    """
    if not body:
        return None

    assigns: list[tuple[str, ast.expr]] = []
    result: ast.expr | None = None
    assign_counts: dict[str, int] = {}

    for i, stmt in enumerate(body):
        is_last = i == len(body) - 1
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                and isinstance(stmt.targets[0], ast.Name):
            name = stmt.targets[0].id
            assign_counts[name] = assign_counts.get(name, 0) + 1
            if is_last:
                result = stmt.value  # trailing assignment → its RHS is the result
            else:
                assigns.append((name, stmt.value))
        elif is_last and isinstance(stmt, ast.Return) and stmt.value is not None:
            result = stmt.value
        elif is_last and isinstance(stmt, ast.Expr):
            result = stmt.value
        else:
            return None  # control flow / unpacking / bare return / etc. → unsafe

    if result is None:
        return None
    # every temporary must be assigned exactly once for an unambiguous substitution
    if any(c != 1 for c in assign_counts.values()):
        return None

    env: dict[str, ast.expr] = {}
    for name, value in assigns:
        env[name] = _substitute(value, env)
    return _substitute(result, env)


def _substitute(node: ast.expr, env: dict[str, ast.expr]) -> ast.expr:
    """Replace every ``Load`` of a name in ``env`` with a copy of its expression."""

    class _Sub(ast.NodeTransformer):
        def visit_Name(self, n: ast.Name) -> ast.AST:
            if isinstance(n.ctx, ast.Load) and n.id in env:
                return copy.deepcopy(env[n.id])
            return n

    return _Sub().visit(copy.deepcopy(node))


class _CommutativeCanon(ast.NodeTransformer):
    """Canonicalise the operand order of the commutative ``Add`` / ``Mult`` ops.

    Bottom-up (children first) so each operand is already canonical before it is
    serialised for ordering.  Only ``Add`` and ``Mult`` are reordered; ``Sub``,
    ``Div``, ``MatMult``, ``Pow``, comparisons and call arguments keep their order.
    """

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, (ast.Add, ast.Mult)):
            left_key = ast.dump(node.left)
            right_key = ast.dump(node.right)
            if right_key < left_key:
                node.left, node.right = node.right, node.left
        return node


def canonical_factor_tree(code: str) -> ast.AST | None:
    """Return the canonicalised AST of ``code``'s factor computation, or ``None``.

    Deterministic and self-contained: two structurally-equivalent computations
    canonicalise to equal (or ``ast.dump``-equal) trees.  ``None`` for empty /
    invalid source.  Numeric literal values appear as the typed placeholder
    sentinels defined in this module.
    """
    extracted = _extract(code)
    if extracted is None:
        return None
    params, body = extracted
    body = _strip_leading_docstring(body)
    body = [s for s in body if not isinstance(s, ast.Pass)]
    if not body:
        return None

    rename = {name: f"_v{i}" for i, name in enumerate(_pre_order_names(params, body))}
    module = ast.Module(body=[copy.deepcopy(s) for s in body], type_ignores=[])
    module = _Canonicaliser(rename).visit(module)
    ast.fix_missing_locations(module)

    inlined = _inline_straightline(list(module.body))
    if inlined is not None:
        module = ast.Module(body=[ast.Return(value=inlined)], type_ignores=[])
    else:
        module = ast.Module(body=[s for s in module.body if s is not None],
                            type_ignores=[])

    module = _CommutativeCanon().visit(module)
    ast.fix_missing_locations(module)
    return module


# ── 3. subtree fingerprints (immutable multiset profile) ─────────────────────

@dataclass(frozen=True)
class SubtreeProfile:
    """Immutable weighted multiset of a program's canonical subtrees.

    ``items`` is a fingerprint→(multiplicity, size) map sorted by fingerprint,
    stored as a tuple of ``(fingerprint_hex, multiplicity, subtree_node_count)``
    triples so the whole profile is hashable / ``lru_cache``-friendly.
    """

    items: tuple[tuple[str, int, int], ...]
    n_nodes: int
    n_unique: int


def _semantic_children(node: ast.AST) -> Iterable[ast.AST]:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _LABEL_ONLY):
            continue  # captured in the parent's label, not a subtree of its own
        yield child


def _node_label(node: ast.AST) -> tuple:
    """Semantic identity of a node: its type + the attributes that carry meaning.

    Operator/attribute/function/field names and string keys are kept; local
    variable names are already alpha-renamed; numeric values are already
    placeholder sentinels.
    """
    t = type(node).__name__
    if isinstance(node, ast.Constant):
        return (t, repr(node.value))
    if isinstance(node, ast.Name):
        return (t, node.id)
    if isinstance(node, ast.Attribute):
        return (t, node.attr)
    if isinstance(node, ast.keyword):
        return (t, node.arg)
    if isinstance(node, ast.BinOp):
        return (t, type(node.op).__name__)
    if isinstance(node, ast.UnaryOp):
        return (t, type(node.op).__name__)
    if isinstance(node, ast.BoolOp):
        return (t, type(node.op).__name__)
    if isinstance(node, ast.Compare):
        return (t, tuple(type(op).__name__ for op in node.ops))
    return (t,)


def _accumulate(node: ast.AST, counts: dict[str, int],
                sizes: dict[str, int]) -> tuple[str, int]:
    """Fold ``node`` into ``counts``/``sizes``; return its ``(fingerprint, size)``."""
    child_fps: list[str] = []
    total = 1
    for child in _semantic_children(node):
        fp, size = _accumulate(child, counts, sizes)
        child_fps.append(fp)
        total += size
    payload = repr((_node_label(node), tuple(child_fps)))
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    counts[fingerprint] = counts.get(fingerprint, 0) + 1
    sizes[fingerprint] = total
    return fingerprint, total


@lru_cache(maxsize=1024)
def subtree_profile(code: str) -> SubtreeProfile | None:
    """Canonical-subtree multiset profile of ``code`` (cached), or ``None``.

    Cached by source string because each archive program is compared against
    many candidates; the cached value is immutable.
    """
    tree = canonical_factor_tree(code)
    if tree is None:
        return None
    counts: dict[str, int] = {}
    sizes: dict[str, int] = {}
    _accumulate(tree, counts, sizes)
    if not counts:
        return None
    items = tuple(sorted((fp, counts[fp], sizes[fp]) for fp in counts))
    return SubtreeProfile(items=items, n_nodes=sum(counts.values()),
                          n_unique=len(counts))


# ── 4. weighted multiset Jaccard similarity / distance ───────────────────────

def _weight(size: int) -> float:
    return 1.0 + math.log(1.0 + size)


def _similarity_from_profiles(pa: SubtreeProfile, pb: SubtreeProfile) -> float | None:
    a = {fp: (mult, size) for fp, mult, size in pa.items}
    b = {fp: (mult, size) for fp, mult, size in pb.items}
    num = 0.0
    den = 0.0
    for fp in set(a) | set(b):
        ma, sa = a.get(fp, (0, 0))
        mb, sb = b.get(fp, (0, 0))
        w = _weight(sa if sa else sb)
        num += w * min(ma, mb)
        den += w * max(ma, mb)
    if den <= 0:
        return None
    return max(0.0, min(1.0, num / den))


def ast_subtree_similarity(code_a: str, code_b: str) -> float | None:
    """Weighted multiset Jaccard similarity in ``[0, 1]`` (``None`` if unmeasurable).

    ``1`` for identical canonical computations, ``0`` for programs with no shared
    canonical subtree.  Symmetric.  ``None`` if either source is empty/invalid.
    """
    pa = subtree_profile(code_a)
    pb = subtree_profile(code_b)
    if pa is None or pb is None:
        return None
    return _similarity_from_profiles(pa, pb)


def ast_subtree_distance(code_a: str, code_b: str) -> float | None:
    """``1 − :func:`ast_subtree_similarity``` — the structural distance in ``[0, 1]``.

    ``0`` = structural clone, ``1`` = maximally different structure.  ``None`` when
    either source is empty/invalid (so the caller keeps "axis unmeasured", never a
    fabricated score).
    """
    sim = ast_subtree_similarity(code_a, code_b)
    return None if sim is None else 1.0 - sim
