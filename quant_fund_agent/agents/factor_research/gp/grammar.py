"""Typed expression-tree grammar for the GP factor-mining benchmark.

A factor is a **typed tree** whose nodes are:

* terminals — a data field (``data["close"]``), or the ``returns`` / ``vwap``
  helper series;
* windows — positive-int lookbacks drawn from a fixed pool (for time-series ops);
* consts — small floats (for ``signed_power`` / ``power`` exponents);
* ops — an operator from the **base grammar** applied to typed children.

Every operator either maps to a named primitive in ``factors.ops`` (whose name
is therefore guaranteed ⊆ :data:`factors.ops.BASE_OPS`) or is one of the four
universal arithmetic operators (``+ - * /``), which are not named ops.  Building
the grammar from this **explicit** table — never ``dir(ops)`` — is what keeps the
GP confined to the base grammar: an op the LLM/framework adds later is invisible
here unless it is deliberately added to both ``BASE_OPS`` and this table.

The tree is deliberately simple and JSON-serialisable so genomes survive a
round-trip through the controller's checkpoint/lineage files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quant_fund_agent.factors.ops import BASE_OPS

# ── type tags ──────────────────────────────────────────────────────────────────
SERIES = "series"     # a (dates × tickers) DataFrame
WINDOW = "window"     # a positive-int lookback
CONST = "const"       # a small float exponent

# Fields that are terminals but rendered via an ``ops`` helper (robust to the
# panel not carrying a literal ``returns`` / ``vwap`` column — the helper
# computes them from close / HLC), and non-numeric fields we never treat as a
# numeric series terminal.
_HELPER_FIELDS = {"returns", "vwap"}
_NON_NUMERIC_FIELDS = {"sector", "industry", "subindustry"}

DEFAULT_WINDOWS: tuple[int, ...] = (2, 3, 5, 10, 15, 20, 30, 40, 60)
DEFAULT_CONSTS: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)


# ── operator table (the base grammar) ──────────────────────────────────────────

@dataclass(frozen=True)
class OpSpec:
    """One operator: its grammar name, typed children, and how it renders.

    ``ops_func`` is the ``factors.ops`` function name (guaranteed in
    ``BASE_OPS``) or ``None`` for a universal arithmetic operator, in which case
    ``symbol`` holds the Python operator.  Every op returns a ``SERIES``.
    """

    name: str
    arg_types: tuple[str, ...]
    ops_func: str | None = None
    symbol: str | None = None
    rtype: str = SERIES


_OPS: tuple[OpSpec, ...] = (
    # ── unary series → series ──
    OpSpec("rank", (SERIES,), "rank"),
    OpSpec("scale", (SERIES,), "scale"),
    OpSpec("log", (SERIES,), "log"),
    OpSpec("abs_", (SERIES,), "abs_"),
    OpSpec("sign", (SERIES,), "sign"),
    # ── series + window → series ──
    OpSpec("delta", (SERIES, WINDOW), "delta"),
    OpSpec("delay", (SERIES, WINDOW), "delay"),
    OpSpec("ts_mean", (SERIES, WINDOW), "ts_mean"),
    OpSpec("ts_sum", (SERIES, WINDOW), "ts_sum"),
    OpSpec("stddev", (SERIES, WINDOW), "stddev"),
    OpSpec("ts_min", (SERIES, WINDOW), "ts_min"),
    OpSpec("ts_max", (SERIES, WINDOW), "ts_max"),
    OpSpec("ts_rank", (SERIES, WINDOW), "ts_rank"),
    OpSpec("ts_zscore", (SERIES, WINDOW), "ts_zscore"),
    OpSpec("decay_linear", (SERIES, WINDOW), "decay_linear"),
    OpSpec("ts_argmax", (SERIES, WINDOW), "ts_argmax"),
    OpSpec("ts_argmin", (SERIES, WINDOW), "ts_argmin"),
    # ── series + const → series ──
    OpSpec("signed_power", (SERIES, CONST), "signed_power"),
    OpSpec("power", (SERIES, CONST), "power"),
    # ── binary arithmetic (universal, not a named op) ──
    OpSpec("add", (SERIES, SERIES), None, "+"),
    OpSpec("sub", (SERIES, SERIES), None, "-"),
    OpSpec("mul", (SERIES, SERIES), None, "*"),
    OpSpec("div", (SERIES, SERIES), None, "/"),
    # ── binary series + window → series ──
    OpSpec("correlation", (SERIES, SERIES, WINDOW), "correlation"),
    OpSpec("covariance", (SERIES, SERIES, WINDOW), "covariance"),
    OpSpec("rolling_beta", (SERIES, SERIES, WINDOW), "rolling_beta"),
    OpSpec("rolling_residual", (SERIES, SERIES, WINDOW), "rolling_residual"),
)

OP_BY_NAME: dict[str, OpSpec] = {op.name: op for op in _OPS}

# Confinement invariant, enforced at import: every named operator this grammar
# can emit is part of the base grammar.  Arithmetic ops (ops_func=None) are the
# four universal Python operators and are not named ops.
_NAMED_GRAMMAR_OPS = frozenset(op.ops_func for op in _OPS if op.ops_func)
assert _NAMED_GRAMMAR_OPS <= BASE_OPS, (
    "GP grammar references ops outside BASE_OPS: "
    f"{sorted(_NAMED_GRAMMAR_OPS - BASE_OPS)}"
)


# ── the tree node ───────────────────────────────────────────────────────────────

@dataclass
class Node:
    """One node in a factor expression tree.

    ``kind`` ∈ {"op", "field", "returns", "vwap", "window", "const"}.  ``value``
    holds the op name / field name / int window / float const as appropriate.
    """

    kind: str
    value: Any = None
    children: tuple["Node", ...] = ()

    @property
    def rtype(self) -> str:
        if self.kind == "window":
            return WINDOW
        if self.kind == "const":
            return CONST
        return SERIES  # op, field, returns, vwap all produce a series

    def copy(self) -> "Node":
        return Node(self.kind, self.value,
                    tuple(c.copy() for c in self.children))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "value": self.value,
                "children": [c.to_dict() for c in self.children]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Node":
        return cls(d["kind"], d.get("value"),
                   tuple(cls.from_dict(c) for c in d.get("children", ())))


# ── the grammar ─────────────────────────────────────────────────────────────────

@dataclass
class Grammar:
    """The operators / terminals available for one run (given the data scope)."""

    field_terminals: tuple[str, ...]      # raw numeric fields → data["X"]
    use_returns: bool                     # the returns(data) helper terminal
    use_vwap: bool                        # the vwap(data) helper terminal
    vwap_inputs: tuple[str, ...]          # fields the vwap terminal needs loaded
    window_pool: tuple[int, ...]
    const_pool: tuple[float, ...]
    ops: tuple[OpSpec, ...] = field(default_factory=lambda: _OPS)

    def terminals(self) -> list[Node]:
        """All series terminals the grammar can emit."""
        out = [Node("field", f) for f in self.field_terminals]
        if self.use_returns:
            out.append(Node("returns"))
        if self.use_vwap:
            out.append(Node("vwap"))
        return out

    def ops_with_signature(self, sig: tuple[str, ...]) -> list[OpSpec]:
        return [op for op in self.ops if op.arg_types == sig]


def build_grammar(
    allowed_fields: list[str] | set[str] | None,
    *,
    window_pool: tuple[int, ...] = DEFAULT_WINDOWS,
    const_pool: tuple[float, ...] = DEFAULT_CONSTS,
    ops: tuple[OpSpec, ...] = _OPS,
) -> Grammar:
    """Build the grammar from the run's in-scope field set.

    Terminals are drawn *only* from ``allowed_fields`` (so every rendered factor
    is data-scope compatible).  ``returns`` / ``vwap`` become helper terminals
    when their underlying fields are in scope.  Raises ``ValueError`` if no
    numeric series field is available (nothing to build a factor from).
    """
    allowed = set(allowed_fields or [])
    raw = tuple(f for f in sorted(allowed)
                if f not in _HELPER_FIELDS and f not in _NON_NUMERIC_FIELDS)
    # Prefer the canonical price/volume fields first (nicer trees), then the rest.
    priority = [f for f in ("close", "open", "high", "low", "volume") if f in raw]
    rest = [f for f in raw if f not in priority]
    field_terminals = tuple(priority + rest)

    use_returns = "close" in allowed or "returns" in allowed
    hlc = tuple(f for f in ("high", "low", "close") if f in allowed)
    # vwap(data) is safe whenever close is available (it falls back to close /
    # (H+L+C)/3); declare exactly the HLC subset in scope as its inputs.
    use_vwap = "vwap" in allowed or "close" in allowed
    vwap_inputs = hlc if hlc else (("close",) if "close" in allowed else ())

    if not field_terminals and not use_returns and not use_vwap:
        raise ValueError(
            "cannot build a GP grammar: no numeric series field in scope "
            f"(allowed_fields={sorted(allowed)})")

    return Grammar(
        field_terminals=field_terminals,
        use_returns=use_returns,
        use_vwap=use_vwap,
        vwap_inputs=vwap_inputs,
        window_pool=tuple(window_pool),
        const_pool=tuple(const_pool),
        ops=tuple(ops),
    )


# ── random tree generation (ramped half-and-half) ───────────────────────────────

def random_tree(
    grammar: Grammar,
    rng: np.random.Generator,
    max_depth: int,
    rtype: str = SERIES,
    *,
    method: str = "grow",
    terminal_prob: float = 0.3,
) -> Node:
    """Grow a random type-valid tree of op-nesting depth ≤ ``max_depth``.

    ``method`` "full" only draws terminals at the depth limit; "grow" may draw a
    terminal earlier (with probability ``terminal_prob``) — ramped half-and-half
    mixes the two across the seed population for size diversity.
    """
    if rtype == WINDOW:
        return Node("window", int(rng.choice(grammar.window_pool)))
    if rtype == CONST:
        return Node("const", float(rng.choice(grammar.const_pool)))

    # rtype == SERIES
    terminals = grammar.terminals()
    force_terminal = max_depth <= 1 or not grammar.ops
    pick_terminal = force_terminal or (
        method == "grow" and float(rng.random()) < terminal_prob)
    if pick_terminal:
        return terminals[int(rng.integers(0, len(terminals)))].copy()

    op = grammar.ops[int(rng.integers(0, len(grammar.ops)))]
    children = tuple(
        random_tree(grammar, rng, max_depth - 1, at, method=method,
                    terminal_prob=terminal_prob)
        for at in op.arg_types
    )
    return Node("op", op.name, children)


# ── tree-structure helpers ──────────────────────────────────────────────────────

def depth(node: Node) -> int:
    """Op-nesting depth: terminals/windows/consts are 0."""
    if not node.children:
        return 0
    return 1 + max(depth(c) for c in node.children)


def size(node: Node) -> int:
    """Total node count."""
    return 1 + sum(size(c) for c in node.children)


def iter_nodes(node: Node, path: tuple[int, ...] = ()) -> list[tuple[tuple[int, ...], Node]]:
    """Depth-first list of ``(path, node)`` — path is the child-index route."""
    out = [(path, node)]
    for i, c in enumerate(node.children):
        out.extend(iter_nodes(c, path + (i,)))
    return out


def get_node(root: Node, path: tuple[int, ...]) -> Node:
    node = root
    for i in path:
        node = node.children[i]
    return node


def replace_at(root: Node, path: tuple[int, ...], new: Node) -> Node:
    """Return a copy of ``root`` with the subtree at ``path`` replaced by ``new``."""
    if not path:
        return new.copy()
    i, rest = path[0], path[1:]
    children = list(root.children)
    children[i] = replace_at(children[i], rest, new)
    return Node(root.kind, root.value, tuple(children))
