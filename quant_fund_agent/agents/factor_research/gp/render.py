"""Render a GP expression tree to a runnable ``BaseFactor`` module.

The output is byte-for-byte a normal factor file — it passes the same
``codegen.validate_code`` the LLM arm's code passes and compiles via
``factors.inmem.compile_factor`` — so a GP factor is scored, persisted and
compared through the identical machinery.  Only the base grammar is used:
imports come exclusively from ``factors.ops`` (whose members here are ⊆
``BASE_OPS``) plus ``numpy`` / ``pandas``.
"""

from __future__ import annotations

import hashlib

from quant_fund_agent.agents.factor_research.evolution.genome import FactorProgram
from quant_fund_agent.agents.factor_research.gp.grammar import (
    OP_BY_NAME,
    Grammar,
    Node,
    iter_nodes,
)


def _fmt_const(value: float) -> str:
    v = float(value)
    return repr(int(v)) + ".0" if v == int(v) else repr(v)


def render_expr(node: Node) -> str:
    """Render one node to a Python expression string."""
    if node.kind == "field":
        return f'data["{node.value}"]'
    if node.kind == "returns":
        return "returns(data)"
    if node.kind == "vwap":
        return "vwap(data)"
    if node.kind == "window":
        return str(int(node.value))
    if node.kind == "const":
        return _fmt_const(node.value)
    if node.kind == "op":
        op = OP_BY_NAME[node.value]
        parts = [render_expr(c) for c in node.children]
        if op.ops_func is None:  # universal arithmetic
            a, b = parts
            if op.symbol == "/":
                # protected division: zeros in the denominator → NaN, never inf
                return f"({a} / ({b}).where(({b}) != 0))"
            return f"({a} {op.symbol} {b})"
        return f"{op.ops_func}({', '.join(parts)})"
    raise ValueError(f"cannot render node kind {node.kind!r}")


def used_ops(tree: Node) -> set[str]:
    """The ``factors.ops`` names the tree calls (for the import line).

    Includes the ``returns`` / ``vwap`` helper terminals.  Every name here is a
    member of ``ops.BASE_OPS`` (the grammar guarantees it), so a test can assert
    the GP never reaches an operator outside the base grammar.
    """
    names: set[str] = set()
    for _, n in iter_nodes(tree):
        if n.kind == "returns":
            names.add("returns")
        elif n.kind == "vwap":
            names.add("vwap")
        elif n.kind == "op":
            spec = OP_BY_NAME[n.value]
            if spec.ops_func:
                names.add(spec.ops_func)
    return names


def used_fields(tree: Node, grammar: Grammar) -> set[str]:
    """The data fields the tree needs loaded (its ``inputs`` list).

    Field terminals contribute themselves; the ``returns`` helper needs
    ``close``; the ``vwap`` helper needs its in-scope HLC subset.
    """
    fields: set[str] = set()
    for _, n in iter_nodes(tree):
        if n.kind == "field":
            fields.add(str(n.value))
        elif n.kind == "returns":
            fields.add("close")
        elif n.kind == "vwap":
            fields.update(grammar.vwap_inputs)
    return fields or {"close"}


def _class_name(expr: str) -> str:
    """A class name derived from the (id-independent) expression.

    Deliberately NOT derived from ``factor_id``: the genome dedup fingerprint
    (``Genome.code_fingerprint``) masks the ``factor_id`` but not the class name,
    so an id-derived class name would make two structurally-identical trees hash
    differently and defeat dedup.  Hashing the expression keeps the class name
    stable across id changes (identical structure → identical masked code).
    """
    return "Gp" + hashlib.sha1(expr.encode()).hexdigest()[:10]


def tree_to_code(
    tree: Node,
    factor_id: str,
    grammar: Grammar,
    *,
    horizon: int,
    category: str = "statistical_arbitrage",
    name: str | None = None,
) -> str:
    """Render the tree to a complete, validator-passing factor module string."""
    ops = sorted(used_ops(tree))
    fields = sorted(used_fields(tree, grammar))
    display = name or factor_id
    import_line = (
        f"from quant_fund_agent.factors.ops import {', '.join(ops)}\n"
        if ops else ""
    )
    inputs_repr = ", ".join(f'"{f}"' for f in fields)
    expr = render_expr(tree)
    class_name = _class_name(expr)
    return (
        f'"""GP-evolved formulaic alpha ({factor_id})."""\n'
        "from __future__ import annotations\n\n"
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "from quant_fund_agent.factors.base import BaseFactor\n"
        "from quant_fund_agent.factors.registry import register_factor\n"
        f"{import_line}\n\n"
        "@register_factor\n"
        f"class {class_name}(BaseFactor):\n"
        f'    factor_id = "{factor_id}"\n'
        f'    name = "{display}"\n'
        f'    category = "{category}"\n'
        f"    inputs = [{inputs_repr}]\n"
        f"    prediction_horizon = {int(horizon)}\n\n"
        "    def calc(self, data):\n"
        f"        return {expr}\n"
    )


def tree_to_program(
    tree: Node,
    factor_id: str,
    grammar: Grammar,
    *,
    horizon: int,
    category: str = "statistical_arbitrage",
    name: str | None = None,
) -> FactorProgram:
    """Render the tree into a :class:`FactorProgram` (code + light metadata)."""
    code = tree_to_code(tree, factor_id, grammar, horizon=horizon,
                        category=category, name=name)
    expr = render_expr(tree)
    return FactorProgram(
        factor_id=factor_id,
        code=code,
        name=name or factor_id,
        category=category,
        trading_idea="GP-evolved formulaic alpha (non-LLM benchmark)",
        description=expr if len(expr) <= 400 else expr[:397] + "...",
        prediction_horizon=int(horizon),
        expected_sign=None,
    )
