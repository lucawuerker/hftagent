"""Non-LLM genetic-programming factor-mining benchmark (AutoAlpha-style).

A deterministic baseline for the LLM evolutionary factor researcher.  It mines
**formulaic alphas** as typed expression trees over the project's *base grammar*
(``factors.ops.BASE_OPS`` + basic arithmetic), renders each tree to a real
``BaseFactor`` module, and scores it with the **exact same** evaluation harness,
NSGA-II controller, Pareto archive and prerun-persistence path the LLM arm uses
(``agents/factor_research/evolution``).  The only thing that differs between the
two arms is *how children are proposed* — LLM mutation vs GP mutation/crossover —
so the comparison is apples-to-apples.

Crucially, the GP is **confined to the base grammar**: its operators are drawn
only from ``ops.BASE_OPS``.  An LLM factor researcher is *not* so confined (it can
define inline helpers and call arbitrary scientific-library primitives) — that
grammar-extension freedom is a deliberate advantage of the agentic framework the
thesis proposes, and it must remain exclusive to the LLM arm.

See ``docs/research-evolution/GP_BENCHMARK.md``.
"""

from __future__ import annotations

from quant_fund_agent.agents.factor_research.gp.grammar import (
    CONST,
    SERIES,
    WINDOW,
    Grammar,
    Node,
    build_grammar,
    depth,
    iter_nodes,
    random_tree,
    size,
)
from quant_fund_agent.agents.factor_research.gp.render import (
    tree_to_code,
    tree_to_program,
    used_fields,
    used_ops,
)

__all__ = [
    "SERIES",
    "WINDOW",
    "CONST",
    "Node",
    "Grammar",
    "build_grammar",
    "random_tree",
    "depth",
    "size",
    "iter_nodes",
    "tree_to_code",
    "tree_to_program",
    "used_fields",
    "used_ops",
]
