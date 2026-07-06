"""Cross-run experience memory (WS5) — survivors **and** controlled negative evidence.

A run should LEARN across runs.  Today the reflection brief, archive and lineage are
per-run and discarded; the only thing that persists is the global GraphRAG semantic
graph.  This module adds a **per-config** experience memory (a
:class:`~quant_fund_agent.knowledge.graph_store.KnowledgeGraph` on its own file) that
accumulates:

* **survivors** — each kept factor's realized performance stamped on its factor node
  (``realized_marginal_value``, ``realized_val_ic``, ``objective``, ``generation``,
  ``verdict``), plus the ``mechanism —realized_by→ factor`` provenance edge; and
* **controlled negative evidence** — per-mechanism/topic ``n_attempts`` / ``n_survived``
  / ``best_marginal`` / ``last_generation`` tallies written for *every scored candidate*,
  not just survivors.  This is the load-bearing bit: survivor-only stamping cannot
  detect exhaustion — a heavily-tried-but-always-failed mechanism would have zero factor
  nodes and look *unexplored*.  The attempt counters make "exhausted" observable
  (many attempts, low survival rate / low mean-|IC|).

The memory feeds the next run's **seeding** (steer away from exhausted mechanisms toward
under-explored ones) and the **teacher** channel (a short summary spliced into the
reflection brief / debate).  Per-config keyed (S&P100 vs LOBSTER never share) via
:func:`memory_graph_path`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from quant_fund_agent.knowledge.empirical_edges import link_factor_to_mechanism
from quant_fund_agent.knowledge.graph_store import KnowledgeGraph, slugify

log = logging.getLogger("knowledge.experience")

MEMORY_ROOT = Path("data/knowledge/memory")


def memory_graph_path(config_name: str) -> Path:
    """Per-config memory file: ``data/knowledge/memory/<config>/graph.json``."""
    return MEMORY_ROOT / slugify(config_name or "default") / "graph.json"


def load_or_new(path: str | Path) -> KnowledgeGraph:
    """Load the memory graph if it exists, else start an empty one (cross-run accrual)."""
    p = Path(path)
    if p.exists():
        try:
            return KnowledgeGraph.load(p)
        except Exception as e:  # noqa: BLE001 — a corrupt memory file must not kill a run
            log.warning("experience memory unreadable (%s) — starting fresh", e)
    return KnowledgeGraph()


# ── writeback ────────────────────────────────────────────────────────────────

def record_attempt(graph: KnowledgeGraph, mechanism_name: str | None, *,
                   survived: bool, marginal: float | None = None,
                   generation: int = 0) -> None:
    """Tally one *scored* candidate against its mechanism — survivors AND failures.

    Idempotency: counters are monotonic within a run; on resume the caller loads the
    saved graph and continues incrementing (a candidate is scored once per run).
    """
    if not mechanism_name:
        return
    mid = graph.add_mechanism(mechanism_name)
    node = graph.g.nodes[mid]
    node["n_attempts"] = int(node.get("n_attempts", 0) or 0) + 1
    if survived:
        node["n_survived"] = int(node.get("n_survived", 0) or 0) + 1
    if marginal is not None:
        prev = node.get("best_marginal")
        node["best_marginal"] = (float(marginal) if prev is None
                                 else max(float(prev), float(marginal)))
    node["last_generation"] = max(int(node.get("last_generation", 0) or 0),
                                  int(generation))


def record_survivor(graph: KnowledgeGraph, factor_id: str, *,
                    objective: dict[str, Any] | None = None,
                    val_ic: float | None = None,
                    marginal_value: float | None = None,
                    generation: int = 0, verdict: str = "kept",
                    mechanism_name: str | None = None) -> None:
    """Stamp a survivor's realized performance onto its factor node (+ provenance)."""
    fnode = graph.add_factor(factor_id)
    attrs = graph.g.nodes[fnode]
    attrs["realized_marginal_value"] = marginal_value
    attrs["realized_val_ic"] = val_ic
    attrs["generation"] = int(generation)
    attrs["verdict"] = verdict
    if objective is not None:
        attrs["objective"] = objective
    if mechanism_name:
        link_factor_to_mechanism(graph, factor_id, mechanism_name)


# ── read (steering + teacher) ────────────────────────────────────────────────

def exhausted_mechanisms(graph: KnowledgeGraph, *, min_attempts: int = 5,
                         max_survival_rate: float = 0.25,
                         max_mean_abs_ic: float | None = None) -> list[str]:
    """Mechanisms tried a lot with little to show → steer the next run away from them.

    "Exhausted" = ``n_attempts ≥ min_attempts`` AND (survival rate ≤
    ``max_survival_rate``, OR ``mean_abs_ic ≤ max_mean_abs_ic`` when provided).
    """
    out: list[str] = []
    for mid in graph.nodes_of_type("mechanism"):
        node = graph.g.nodes[mid]
        n = int(node.get("n_attempts", 0) or 0)
        if n < min_attempts:
            continue
        rate = (int(node.get("n_survived", 0) or 0) / n) if n else 0.0
        flagged = rate <= max_survival_rate
        if max_mean_abs_ic is not None and node.get("mean_abs_ic") is not None:
            flagged = flagged or float(node["mean_abs_ic"]) <= max_mean_abs_ic
        if flagged:
            out.append(node.get("name", mid.split(":", 1)[1]))
    return sorted(out)


def memory_summary(graph: KnowledgeGraph, *, top_k: int = 5) -> str:
    """A short natural-language memory brief for the teacher channel.

    Empty string when the memory holds nothing yet (so callers can splice
    unconditionally).
    """
    mechs = graph.nodes_of_type("mechanism")
    tried = [m for m in mechs if int(graph.g.nodes[m].get("n_attempts", 0) or 0) > 0]
    if not tried:
        return ""
    exhausted = exhausted_mechanisms(graph)
    # best survivors by realized marginal value
    survivors = [
        (graph.g.nodes[f].get("name", f.split(":", 1)[1]),
         graph.g.nodes[f].get("realized_marginal_value"))
        for f in graph.nodes_of_type("factor")
        if graph.g.nodes[f].get("realized_marginal_value") is not None]
    survivors.sort(key=lambda t: (t[1] if t[1] is not None else -1e9), reverse=True)
    parts = [f"EXPERIENCE MEMORY: {len(tried)} mechanism(s) tried across prior runs."]
    if exhausted:
        parts.append("Exhausted (avoid — many attempts, little edge): "
                     + ", ".join(exhausted[:top_k]) + ".")
    if survivors:
        top = ", ".join(f"{n} ({v:+.3f})" for n, v in survivors[:top_k])
        parts.append(f"Best prior factors: {top}.")
    return " ".join(parts)
