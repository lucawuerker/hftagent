"""Typed genetic operators for the GP factor miner.

Standard tree GP — subtree crossover, subtree mutation, point mutation, hoist
mutation — but **type-aware**: a crosspoint only accepts a donor of the same
node type (a window subtree never lands where a series is expected), and point
mutation only swaps an operator for one with an identical argument signature.
Every operator returns a fresh, type-valid, ``SERIES``-rooted tree whose op-depth
respects ``max_depth`` (checked explicitly, with bounded retries).

These replace the LLM mutation/crossover operators; everything downstream (render
→ compile → evaluate → NSGA-II insert) is unchanged.
"""

from __future__ import annotations

import numpy as np

from quant_fund_agent.agents.factor_research.gp.grammar import (
    OP_BY_NAME,
    SERIES,
    Grammar,
    Node,
    depth,
    get_node,
    iter_nodes,
    random_tree,
    replace_at,
)

_MAX_TRIES = 8


def _random_terminal(grammar: Grammar, rng: np.random.Generator) -> Node:
    terms = grammar.terminals()
    return terms[int(rng.integers(0, len(terms)))].copy()


def subtree_crossover(
    a: Node,
    b: Node,
    grammar: Grammar,
    rng: np.random.Generator,
    max_depth: int,
) -> Node:
    """Swap a random subtree of ``a`` with a same-typed subtree of ``b``.

    Biased toward series crosspoints (the meaningful recombination); falls back
    to a copy of ``a`` if no depth-respecting swap is found.
    """
    b_by_type: dict[str, list[Node]] = {}
    for _, n in iter_nodes(b):
        b_by_type.setdefault(n.rtype, []).append(n)

    a_nodes = [(p, n) for p, n in iter_nodes(a) if n.rtype in b_by_type]
    if not a_nodes:
        return a.copy()
    series_nodes = [(p, n) for p, n in a_nodes if n.rtype == SERIES]

    for _ in range(_MAX_TRIES):
        pool = series_nodes if series_nodes and float(rng.random()) < 0.9 else a_nodes
        p, n = pool[int(rng.integers(0, len(pool)))]
        donors = b_by_type[n.rtype]
        donor = donors[int(rng.integers(0, len(donors)))]
        child = replace_at(a, p, donor)
        if depth(child) <= max_depth:
            return child
    return a.copy()


def subtree_mutation(
    tree: Node,
    grammar: Grammar,
    rng: np.random.Generator,
    max_depth: int,
) -> Node:
    """Replace a random subtree with a fresh random tree of the same type."""
    nodes = iter_nodes(tree)
    for _ in range(_MAX_TRIES):
        p, n = nodes[int(rng.integers(0, len(nodes)))]
        budget = max(1, max_depth - len(p))
        new_sub = random_tree(grammar, rng, budget, n.rtype, method="grow")
        child = replace_at(tree, p, new_sub)
        if depth(child) <= max_depth:
            return child
    return tree.copy()


def point_mutation(
    tree: Node,
    grammar: Grammar,
    rng: np.random.Generator,
) -> Node:
    """Swap one node for a compatible one (op of same signature, or a leaf)."""
    nodes = iter_nodes(tree)
    p, n = nodes[int(rng.integers(0, len(nodes)))]
    return replace_at(tree, p, _mutate_point(n, grammar, rng))


def _mutate_point(n: Node, grammar: Grammar, rng: np.random.Generator) -> Node:
    if n.kind == "window":
        return Node("window", int(rng.choice(grammar.window_pool)))
    if n.kind == "const":
        return Node("const", float(rng.choice(grammar.const_pool)))
    if n.kind in ("field", "returns", "vwap"):
        return _random_terminal(grammar, rng)
    if n.kind == "op":
        op = OP_BY_NAME[n.value]
        alts = [o for o in grammar.ops_with_signature(op.arg_types)
                if o.name != op.name]
        if alts:
            new_op = alts[int(rng.integers(0, len(alts)))]
            return Node("op", new_op.name, n.children)  # same signature → keep children
        return n.copy()
    return n.copy()


def hoist_mutation(tree: Node, rng: np.random.Generator) -> Node:
    """Replace the tree with one of its proper series subtrees (parsimony)."""
    series_subtrees = [
        (p, n) for p, n in iter_nodes(tree) if n.rtype == SERIES and p != ()
    ]
    if not series_subtrees:
        return tree.copy()
    _, n = series_subtrees[int(rng.integers(0, len(series_subtrees)))]
    return n.copy()
