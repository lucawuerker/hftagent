"""The custom hybrid knowledge graph: schema + persistence (NetworkX).

The novelty (design §GraphRAG): **fuse an LLM-extracted semantic graph with a
computed quantitative graph** in one store.

Node types (namespaced ids so one graph holds them all):

* ``paper:<id>``      — a paper from the library.
* ``mechanism:<slug>`` — an economic mechanism (LLM-extracted, canonicalised).
* ``factor:<id>``     — a factor program (seed / researched / evolved).
* ``field:<name>``    — a data field (``close``, ``orderFlow``, ``peRatio``, …).
* ``anomaly:<slug>``  — a named market anomaly (LLM-extracted).

Edge relations (edge attr ``relation``):

* semantic (LLM-extracted): ``proposes`` (paper→mechanism), ``realized_by``
  (mechanism→factor), ``related_to`` (mechanism↔mechanism), ``documents``
  (paper→anomaly), ``needs`` (mechanism→field — the fields a mechanism
  plausibly requires, which powers "computable but unexploited" gap-finding);
* empirical (computed, refreshed each generation — see
  :mod:`~quant_fund_agent.knowledge.empirical_edges`): ``corr``
  (factor↔factor, ``weight`` = signal correlation), ``uses`` (factor→field).

Mechanism *coverage* (how many factors realise it, at what IC) is stored as
node attributes (``n_factors``, ``mean_abs_ic``) rather than an edge.

Community detection: NetworkX's built-in **Louvain** (Leiden's direct
predecessor; ``leidenalg`` is used instead when installed — the two agree on
graphs this size, and Louvain keeps the dependency footprint pure-Python).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

log = logging.getLogger("knowledge.graph_store")

DEFAULT_GRAPH_PATH = Path("data/knowledge/graph.json")

SEMANTIC_RELATIONS = ("proposes", "realized_by", "related_to", "documents", "needs")
EMPIRICAL_RELATIONS = ("corr", "uses")


def slugify(name: str) -> str:
    """Canonical mechanism/anomaly slug: lowercase, underscores, trimmed."""
    import re

    s = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    # crude singularisation so "momentum spillovers" ≡ "momentum spillover"
    if s.endswith("s") and not s.endswith("ss") and len(s) > 3:
        s = s[:-1]
    return s[:80]


class KnowledgeGraph:
    """Typed façade over one ``nx.DiGraph`` (semantic + empirical layers)."""

    def __init__(self, graph: nx.DiGraph | None = None) -> None:
        self.g = graph if graph is not None else nx.DiGraph()

    # ── node helpers ──

    @staticmethod
    def paper_id(pid: str) -> str:
        return f"paper:{pid}"

    @staticmethod
    def mechanism_id(name: str) -> str:
        return f"mechanism:{slugify(name)}"

    @staticmethod
    def factor_id(fid: str) -> str:
        return f"factor:{fid}"

    @staticmethod
    def field_id(name: str) -> str:
        return f"field:{name}"

    @staticmethod
    def anomaly_id(name: str) -> str:
        return f"anomaly:{slugify(name)}"

    def add_paper(self, pid: str, *, title: str = "",
                  published_date: str | None = None) -> str:
        nid = self.paper_id(pid)
        self.g.add_node(nid, type="paper", title=title,
                        published_date=published_date)
        return nid

    def add_mechanism(self, name: str, *, description: str = "") -> str:
        nid = self.mechanism_id(name)
        if nid in self.g and not description:
            return nid
        prev = self.g.nodes[nid].get("description", "") if nid in self.g else ""
        self.g.add_node(nid, type="mechanism", name=name,
                        description=description or prev)
        return nid

    def add_factor(self, fid: str, *, name: str = "", category: str = "") -> str:
        nid = self.factor_id(fid)
        self.g.add_node(nid, type="factor", name=name or fid, category=category)
        return nid

    def add_field(self, name: str) -> str:
        nid = self.field_id(name)
        self.g.add_node(nid, type="field", name=name)
        return nid

    def add_anomaly(self, name: str, *, description: str = "") -> str:
        nid = self.anomaly_id(name)
        self.g.add_node(nid, type="anomaly", name=name, description=description)
        return nid

    def add_relation(self, src: str, dst: str, relation: str, **attrs: Any) -> None:
        if relation not in SEMANTIC_RELATIONS + EMPIRICAL_RELATIONS:
            raise ValueError(f"unknown relation {relation!r}")
        self.g.add_edge(src, dst, relation=relation, **attrs)

    # ── typed views ──

    def nodes_of_type(self, node_type: str) -> list[str]:
        return [n for n, d in self.g.nodes(data=True) if d.get("type") == node_type]

    def neighbors_by_relation(self, node: str, relation: str,
                              *, incoming: bool = False) -> list[str]:
        if node not in self.g:
            return []
        if incoming:
            return [u for u, _, d in self.g.in_edges(node, data=True)
                    if d.get("relation") == relation]
        return [v for _, v, d in self.g.out_edges(node, data=True)
                if d.get("relation") == relation]

    def papers_for_mechanism(self, mech: str) -> list[str]:
        return self.neighbors_by_relation(mech, "proposes", incoming=True)

    def factors_for_mechanism(self, mech: str) -> list[str]:
        return self.neighbors_by_relation(mech, "realized_by")

    def fields_needed(self, mech: str) -> list[str]:
        return self.neighbors_by_relation(mech, "needs")

    # ── communities ──

    def detect_communities(self, seed: int = 0) -> dict[str, int]:
        """Community id per node over the undirected semantic projection.

        Uses ``leidenalg`` when installed, else NetworkX Louvain.  Only the
        *semantic* layer participates (papers, mechanisms, anomalies + their
        edges): empirical corr edges reflect a different similarity notion and
        would smear the mechanism-level structure the islands want to target.
        Community ids are stamped onto the nodes (attr ``community``).
        """
        semantic_nodes = {n for n, d in self.g.nodes(data=True)
                          if d.get("type") in ("paper", "mechanism", "anomaly")}
        sub = self.g.edge_subgraph([
            (u, v) for u, v, d in self.g.edges(data=True)
            if d.get("relation") in SEMANTIC_RELATIONS
            and u in semantic_nodes and v in semantic_nodes
        ]).to_undirected() if self.g.edges else nx.Graph()

        # isolated semantic nodes still deserve a community
        graph = nx.Graph()
        graph.add_nodes_from(semantic_nodes)
        graph.add_edges_from(sub.edges())

        communities: list[set[str]]
        try:
            import igraph  # noqa: F401
            import leidenalg  # noqa: F401

            communities = self._leiden(graph, seed)
        except ImportError:
            from networkx.algorithms.community import louvain_communities

            communities = louvain_communities(graph, seed=seed) if len(graph) else []

        assignment: dict[str, int] = {}
        for ci, members in enumerate(sorted(communities, key=len, reverse=True)):
            for n in members:
                assignment[n] = ci
                self.g.nodes[n]["community"] = ci
        return assignment

    @staticmethod
    def _leiden(graph: nx.Graph, seed: int) -> list[set[str]]:  # pragma: no cover
        import igraph as ig
        import leidenalg

        nodes = list(graph.nodes())
        idx = {n: i for i, n in enumerate(nodes)}
        g = ig.Graph(n=len(nodes),
                     edges=[(idx[u], idx[v]) for u, v in graph.edges()])
        part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition,
                                        seed=seed)
        return [{nodes[i] for i in comm} for comm in part]

    def community_members(self) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        for n, d in self.g.nodes(data=True):
            c = d.get("community")
            if c is not None:
                out.setdefault(int(c), []).append(n)
        return out

    # ── persistence ──

    def save(self, path: str | Path = DEFAULT_GRAPH_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = nx.node_link_data(self.g, edges="edges")
        path.write_text(json.dumps(payload, indent=1, default=str))
        log.info("saved knowledge graph (%d nodes, %d edges) → %s",
                 self.g.number_of_nodes(), self.g.number_of_edges(), path)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_GRAPH_PATH) -> "KnowledgeGraph":
        payload = json.loads(Path(path).read_text())
        return cls(nx.node_link_graph(payload, directed=True, edges="edges"))

    # ── stats ──

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, d in self.g.nodes(data=True):
            t = d.get("type", "?")
            counts[t] = counts.get(t, 0) + 1
        counts["edges"] = self.g.number_of_edges()
        return counts
