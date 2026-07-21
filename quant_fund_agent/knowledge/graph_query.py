"""Query API over the hybrid graph: global gap-finding + local grounding.

Two query families (design §GraphRAG):

* **Global / gap-finding** — where should the search look next?
  :func:`mechanism_gaps` (papers propose it, no factor realises it),
  :func:`computable_unexploited` (gaps whose required fields the current run
  can actually serve — novel AND computable), :func:`under_covered_communities`
  (whole research areas the book underweights) and :func:`island_focus`
  (per-island steering strings, giving the islands mechanism-level diversity).
* **Local / grounding** — :func:`local_context`: everything attached to one
  mechanism (papers, realising factors, related mechanisms, needed fields) —
  the natural multi-paper ``N→1`` synthesis bundle, and the provenance path
  ``Paper → Mechanism → Factor → Field`` for the thesis narrative.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from quant_fund_agent.knowledge.graph_store import KnowledgeGraph

log = logging.getLogger("knowledge.graph_query")


def mechanism_gaps(graph: KnowledgeGraph, *, min_papers: int = 1) -> list[str]:
    """Mechanisms with ≥ ``min_papers`` proposing papers and NO realising factor,
    ordered by paper support (most-documented gaps first)."""
    gaps: list[tuple[int, str]] = []
    for mech in graph.nodes_of_type("mechanism"):
        n_papers = len(graph.papers_for_mechanism(mech))
        n_factors = len(graph.factors_for_mechanism(mech))
        if n_papers >= min_papers and n_factors == 0:
            gaps.append((n_papers, mech))
    # most-documented first; ties broken by id so the order is stable run-to-run
    return [m for _, m in sorted(gaps, key=lambda t: (-t[0], t[1]))]


def computable_unexploited(graph: KnowledgeGraph,
                           available_fields: Sequence[str],
                           *, min_papers: int = 1) -> list[str]:
    """Gap mechanisms whose declared field needs the current run can serve.

    A mechanism with no ``needs`` edges is kept (nothing known to block it);
    one needing an out-of-scope field is dropped — "novel + computable" is the
    combination that makes a gap worth an island.
    """
    available = set(available_fields)
    out: list[str] = []
    for mech in mechanism_gaps(graph, min_papers=min_papers):
        needed = {n.split(":", 1)[1] for n in graph.fields_needed(mech)}
        if needed <= available:
            out.append(mech)
    return out


def under_covered_communities(graph: KnowledgeGraph, *, k: int = 3,
                              ) -> list[tuple[int, float]]:
    """Communities ranked by factor coverage per paper, thinnest first.

    Returns ``[(community_id, coverage_ratio), ...]`` where the ratio is
    ``#factors attached to the community's mechanisms / #papers in it`` — a
    whole research area with many papers and few factors is where structured
    gap-finding beats flat similarity retrieval.
    """
    scores: list[tuple[float, int]] = []
    for cid, members in graph.community_members().items():
        mechs = [m for m in members if graph.g.nodes[m].get("type") == "mechanism"]
        papers = [m for m in members if graph.g.nodes[m].get("type") == "paper"]
        if not mechs or not papers:
            continue
        n_factors = sum(len(graph.factors_for_mechanism(m)) for m in mechs)
        scores.append((n_factors / max(1, len(papers)), cid))
    scores.sort()
    return [(cid, ratio) for ratio, cid in scores[:k]]


def local_context(graph: KnowledgeGraph, mechanism: str) -> dict[str, Any]:
    """Everything attached to one mechanism — the grounding bundle.

    ``mechanism`` may be a node id (``mechanism:slug``) or a raw name.
    """
    mid = mechanism if mechanism.startswith("mechanism:") \
        else graph.mechanism_id(mechanism)
    if mid not in graph.g:
        return {}
    node = graph.g.nodes[mid]
    papers = [{
        "paper_id": p.split(":", 1)[1],
        "title": graph.g.nodes[p].get("title", ""),
        "published_date": graph.g.nodes[p].get("published_date"),
    } for p in graph.papers_for_mechanism(mid)]
    return {
        "mechanism": node.get("name", mid),
        "description": node.get("description", ""),
        "community": node.get("community"),
        "n_factors": node.get("n_factors", len(graph.factors_for_mechanism(mid))),
        "mean_abs_ic": node.get("mean_abs_ic"),
        "papers": papers,
        "factors": [f.split(":", 1)[1] for f in graph.factors_for_mechanism(mid)],
        "related_mechanisms": [
            graph.g.nodes[r].get("name", r)
            for r in graph.neighbors_by_relation(mid, "related_to")],
        "needed_fields": [n.split(":", 1)[1] for n in graph.fields_needed(mid)],
    }


def mechanism_group_specs(
    graph: KnowledgeGraph,
    n_groups: int,
    available_fields: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Build reserved knowledge-graph mechanism groups for one evolution run.

    Groups use the thinnest communities' summaries plus their most
    paper-supported computable gap mechanisms — structural, mechanism-level
    coverage.  The returned mechanism names are used for retrieval tagging and the
    stable integer id is written into every descendant genome.
    """
    summaries = graph.g.graph.get("community_summaries", {})
    gaps = (computable_unexploited(graph, available_fields)
            if available_fields is not None else mechanism_gaps(graph))
    gap_names = [graph.g.nodes[m].get("name", m) for m in gaps]

    specs: list[dict[str, Any]] = []
    ranked = under_covered_communities(graph, k=max(n_groups, 1))
    for cid, _ratio in ranked:
        members = graph.community_members().get(cid, [])
        community_gaps = [graph.g.nodes[m].get("name", m) for m in gaps
                          if m in members]
        parts = [summaries.get(str(cid), "")]
        if community_gaps:
            parts.append("target mechanisms: " + ", ".join(community_gaps[:4]))
        specs.append({
            "mechanism_group_id": len(specs),
            "community_id": cid,
            "focus": " ".join(p for p in parts if p).strip(),
            "mechanisms": community_gaps[:4],
        })
        if len(specs) >= n_groups:
            break

    # top up from the flat gap list when communities are missing/too few
    gi = 0
    while len(specs) < n_groups:
        chunk = gap_names[gi:gi + 3]
        specs.append({
            "mechanism_group_id": len(specs),
            "community_id": None,
            "focus": "target mechanisms: " + ", ".join(chunk) if chunk else "",
            "mechanisms": chunk,
        })
        gi += 3
        if gi > len(gap_names) + 3:
            break
    return specs[:n_groups]


def island_focus(graph: KnowledgeGraph, n_islands: int,
                 available_fields: Sequence[str] | None = None) -> list[str]:
    """Backward-compatible focus-only view of :func:`mechanism_group_specs`."""
    return [s["focus"] for s in mechanism_group_specs(
        graph, n_islands, available_fields)]
