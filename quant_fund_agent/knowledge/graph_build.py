"""LLM extraction pass: papers → (mechanisms, anomalies, field needs) → the graph.

One structured LLM call per paper (idempotent: already-ingested papers are
skipped, so the build is resumable and cheap to extend as the library grows).
Mechanism names are canonicalised by slug (see
:func:`~quant_fund_agent.knowledge.graph_store.slugify`) so "momentum
spillovers" and "Momentum spillover" merge into one node — the
canonicalisation/merge pass the design flags as the main extraction-noise risk;
`scripts/build_knowledge_graph.py` prints the mechanism list for the human
spot-check it recommends.

After extraction, :func:`detect_and_summarize_communities` runs community
detection and writes a deterministic summary per community (member mechanisms +
paper counts).  Pass an LLM to get prose summaries instead — the deterministic
fallback keeps tests and offline builds working.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Sequence

from quant_fund_agent.knowledge.embed_store import CorpusDoc
from quant_fund_agent.knowledge.graph_store import KnowledgeGraph

log = logging.getLogger("knowledge.graph_build")

EXTRACTION_PROMPT = """\
You are building a knowledge graph over quantitative-finance research.  Read
the paper below and extract its *economic mechanisms* — the causal stories
about why a return pattern exists (e.g. "underreaction to earnings news",
"inventory risk premium of market makers"), NOT the formulas.

For each mechanism, also list:
- the DATA FIELDS one would plausibly need to exploit it, chosen ONLY from:
  {field_vocab}
- any NAMED market anomaly the paper ties it to (e.g. "post-earnings
  announcement drift", "momentum crash"), if one is named.
- related mechanisms *from this same paper*, if it explicitly links them.

Extract 1-4 mechanisms; fewer, well-named mechanisms beat many vague ones.
Use short, canonical, reusable names ("order flow imbalance pressure", not
"the effect we document in section 4.2").

PAPER (id={paper_id})
---------------------
{title}

{text}

Respond with strict JSON:
{{
  "mechanisms": [
    {{
      "name": "<short canonical mechanism name>",
      "description": "<1-2 sentences: the causal story>",
      "required_fields": ["<field>", ...],
      "anomaly": "<named anomaly or null>",
      "related_mechanisms": ["<other mechanism name from this paper>", ...]
    }}
  ]
}}
"""

COMMUNITY_SUMMARY_PROMPT = """\
Summarise this cluster of related quantitative-finance research in 2-3
sentences: what family of mechanisms is it about, and what kind of alpha
factors would exploit it?

MECHANISMS
----------
{mechanisms}

PAPERS
------
{papers}

Respond with strict JSON: {{"summary": "<2-3 sentences>"}}
"""


def _invoke_json(llm: Any, prompt: str) -> dict:
    from quant_fund_agent.agents.factor_research.graph import _parse_json

    resp = llm.invoke(prompt)
    return _parse_json(getattr(resp, "content", str(resp)))


def extract_paper(llm: Any, doc: CorpusDoc, field_vocab: Sequence[str],
                  max_chars: int = 24_000) -> dict:
    """One extraction call for one paper → the raw mechanisms payload."""
    prompt = EXTRACTION_PROMPT.format(
        field_vocab=", ".join(sorted(field_vocab)),
        paper_id=doc.paper_id, title=doc.title,
        text=(doc.text or "")[:max_chars])
    return _invoke_json(llm, prompt)


def ingest_paper(graph: KnowledgeGraph, doc: CorpusDoc, payload: dict,
                 field_vocab: Sequence[str]) -> int:
    """Fold one paper's extraction payload into the graph → #mechanisms added."""
    pid = graph.add_paper(doc.paper_id, title=doc.title,
                          published_date=doc.published_date)
    vocab = set(field_vocab)
    n = 0
    mechanisms = payload.get("mechanisms") or []
    for mech in mechanisms:
        name = (mech.get("name") or "").strip()
        if not name:
            continue
        mid = graph.add_mechanism(name, description=mech.get("description", ""))
        graph.add_relation(pid, mid, "proposes")
        n += 1
        for f in mech.get("required_fields") or []:
            if f in vocab:  # hallucinated fields never enter the graph
                graph.add_relation(mid, graph.add_field(f), "needs")
        anomaly = (mech.get("anomaly") or "").strip()
        if anomaly and anomaly.lower() not in ("null", "none"):
            aid = graph.add_anomaly(anomaly)
            graph.add_relation(pid, aid, "documents")
        for rel in mech.get("related_mechanisms") or []:
            rel = (rel or "").strip()
            if rel and rel.lower() != name.lower():
                rid = graph.add_mechanism(rel)
                graph.add_relation(mid, rid, "related_to")
    graph.g.nodes[pid]["ingested"] = True
    return n


def build_graph(
    llm: Any,
    docs: Sequence[CorpusDoc],
    *,
    graph: KnowledgeGraph | None = None,
    field_vocab: Sequence[str] | None = None,
    max_papers: int | None = None,
) -> KnowledgeGraph:
    """Extraction pass over ``docs`` (skipping already-ingested papers).

    ``field_vocab`` defaults to the run's usable fields so the ``needs`` edges
    stay within what any run could actually serve.  Failures on single papers
    are logged and skipped — a 1k-paper build must survive flaky calls.
    """
    graph = graph or KnowledgeGraph()
    if field_vocab is None:
        try:
            from quant_fund_agent.data import usable_fields

            field_vocab = sorted(usable_fields())
        except Exception:  # noqa: BLE001
            field_vocab = ["open", "high", "low", "close", "volume"]

    done = 0
    for doc in docs:
        if max_papers is not None and done >= max_papers:
            break
        nid = graph.paper_id(doc.paper_id)
        if nid in graph.g and graph.g.nodes[nid].get("ingested"):
            continue
        try:
            payload = extract_paper(llm, doc, field_vocab)
            n = ingest_paper(graph, doc, payload, field_vocab)
            done += 1
            log.info("[graph_build] %s → %d mechanism(s)", doc.paper_id, n)
        except Exception as e:  # noqa: BLE001 — one flaky paper must not kill the build
            log.warning("[graph_build] %s failed: %s", doc.paper_id, e)
    return graph


def detect_and_summarize_communities(graph: KnowledgeGraph,
                                     llm: Any | None = None,
                                     seed: int = 0) -> dict[int, str]:
    """Community detection + one summary per community (stored on the graph).

    Deterministic summaries (mechanism names + counts) unless an ``llm`` is
    given.  Returns ``{community_id: summary}`` and stamps each mechanism node's
    community; summaries live in ``graph.g.graph['community_summaries']``.
    """
    graph.detect_communities(seed=seed)
    summaries: dict[int, str] = {}
    for cid, members in sorted(graph.community_members().items()):
        mechs = [graph.g.nodes[m].get("name", m) for m in members
                 if graph.g.nodes[m].get("type") == "mechanism"]
        papers = [graph.g.nodes[m].get("title", m) for m in members
                  if graph.g.nodes[m].get("type") == "paper"]
        if llm is not None and mechs:
            try:
                payload = _invoke_json(llm, COMMUNITY_SUMMARY_PROMPT.format(
                    mechanisms="\n".join(f"- {m}" for m in mechs),
                    papers="\n".join(f"- {p}" for p in papers[:20])))
                summaries[cid] = str(payload.get("summary", ""))
                continue
            except Exception as e:  # noqa: BLE001 — deterministic fallback below
                log.warning("community %d summary call failed: %s", cid, e)
        summaries[cid] = (
            f"Community {cid}: {len(mechs)} mechanism(s) "
            f"({', '.join(sorted(mechs)[:6])}{'…' if len(mechs) > 6 else ''}) "
            f"across {len(papers)} paper(s).")
    graph.g.graph["community_summaries"] = {str(k): v for k, v in summaries.items()}
    return summaries
