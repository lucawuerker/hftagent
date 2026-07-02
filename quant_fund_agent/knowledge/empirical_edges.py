"""The computed (quantitative) layer of the hybrid graph — refreshed each generation.

Three refreshers, all deterministic:

* :func:`refresh_factor_correlations` — ``factor —corr→ factor`` edges from the
  factors' *signal* correlation (per-underlying z-scored, flattened, pairwise
  over finite cells; only |corr| ≥ threshold edges are kept so the graph stays
  sparse).  This is what lets the debate skeptic and the gap-finder see
  *numeric* redundancy next to semantic relatedness — the fusion that makes the
  hybrid graph worth owning.
* :func:`refresh_field_usage` — ``factor —uses→ field`` edges from each
  factor's declared ``inputs``.
* :func:`refresh_mechanism_coverage` — per-mechanism coverage stats
  (``n_factors``, ``mean_abs_ic``) rolled up from the factors that realise it,
  stored as node attributes.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from quant_fund_agent.knowledge.graph_store import KnowledgeGraph

log = logging.getLogger("knowledge.empirical_edges")


def refresh_factor_correlations(
    graph: KnowledgeGraph,
    signals: Mapping[str, pd.DataFrame],
    *,
    threshold: float = 0.3,
) -> int:
    """Recompute the factor↔factor ``corr`` edges from signal frames.

    Existing ``corr`` edges are dropped first (this layer is *derived* state —
    it must always reflect the latest signals, never accrete).  Returns the
    number of edges written.  Signals are z-scored per underlying and flattened
    so the correlation is over (date, ticker) cells, matching the independence
    axis in the fitness harness.
    """
    from quant_fund_agent.comparison.standardize import per_underlying_zscore

    stale = [(u, v) for u, v, d in graph.g.edges(data=True)
             if d.get("relation") == "corr"]
    graph.g.remove_edges_from(stale)

    ids = list(signals)
    flat: dict[str, np.ndarray] = {}
    for fid in ids:
        z = per_underlying_zscore(signals[fid])
        flat[fid] = z.to_numpy(dtype=float).ravel()

    n_edges = 0
    for i, a in enumerate(ids):
        graph.add_factor(a)
        for b in ids[i + 1:]:
            va, vb = flat[a], flat[b]
            m = min(len(va), len(vb))
            mask = np.isfinite(va[:m]) & np.isfinite(vb[:m])
            if mask.sum() < 30:
                continue
            corr = float(np.corrcoef(va[:m][mask], vb[:m][mask])[0, 1])
            if np.isfinite(corr) and abs(corr) >= threshold:
                graph.add_factor(b)
                graph.add_relation(graph.factor_id(a), graph.factor_id(b),
                                   "corr", weight=round(corr, 4))
                n_edges += 1
    log.info("refreshed corr edges: %d (|corr| ≥ %.2f) over %d factors",
             n_edges, threshold, len(ids))
    return n_edges


def refresh_field_usage(graph: KnowledgeGraph,
                        factor_inputs: Mapping[str, Sequence[str]]) -> int:
    """Recompute ``factor —uses→ field`` edges from declared inputs."""
    stale = [(u, v) for u, v, d in graph.g.edges(data=True)
             if d.get("relation") == "uses"]
    graph.g.remove_edges_from(stale)

    n = 0
    for fid, inputs in factor_inputs.items():
        fnode = graph.add_factor(fid)
        for f in inputs or []:
            graph.add_relation(fnode, graph.add_field(f), "uses")
            n += 1
    return n


def link_factor_to_mechanism(graph: KnowledgeGraph, factor_id: str,
                             mechanism_name: str) -> None:
    """Record that a factor realises a mechanism (called when a factor is born
    from a graph-grounded idea, closing the Paper → Mechanism → Factor → Field
    provenance path)."""
    mid = graph.add_mechanism(mechanism_name)
    fid = graph.add_factor(factor_id)
    graph.add_relation(mid, fid, "realized_by")


def refresh_mechanism_coverage(graph: KnowledgeGraph,
                               factor_ics: Mapping[str, float | None]) -> None:
    """Roll factor ICs up to mechanism coverage stats (node attrs).

    ``factor_ics`` maps factor id → its recorded IC (``None`` allowed).  Every
    mechanism gets ``n_factors`` and ``mean_abs_ic`` refreshed; mechanisms with
    no realising factors get ``n_factors=0`` — exactly the "papers but no
    factor yet" gap the global query hunts for.
    """
    for mech in graph.nodes_of_type("mechanism"):
        fids = [n.split(":", 1)[1] for n in graph.factors_for_mechanism(mech)]
        ics = [abs(factor_ics[f]) for f in fids
               if f in factor_ics and factor_ics[f] is not None]
        graph.g.nodes[mech]["n_factors"] = len(fids)
        graph.g.nodes[mech]["mean_abs_ic"] = (
            round(float(np.mean(ics)), 6) if ics else None)
