"""Retrieval-grounded idea generation: query construction + cardinality modes.

This is the seam between the :class:`~quant_fund_agent.knowledge.embed_store.
EmbedStore` and the brainstorm LLM.  Three cardinality modes (design §RAG):

* ``1toN`` — today's shape: each retrieved paper grounds its own call, ideas
  are pooled (idea provenance = that one paper, attributed authoritatively).
* ``Nto1`` — the synthesis mode: N papers in ONE call, and the LLM must fuse
  mechanisms *across* papers into each idea (where genuinely novel factors come
  from).  Budget is asked one idea at a time in spirit but batched as one call
  asking for the full budget with a cross-paper synthesis requirement.
* ``NtoM`` — N papers, M ideas, no synthesis constraint (the LLM may ground an
  idea in one paper or several).

Every mode date-gates retrieval and **verifies citations**: a claimed
``source_paper_id`` outside the retrieved set is a hallucination — the citation
is dropped, and an idea left with no verified grounding in the synthesis modes
is discarded.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from quant_fund_agent.knowledge.embed_store import (
    EmbedStore,
    RetrievedPaper,
    verify_citations,
)

log = logging.getLogger("knowledge.retrieval")

CARDINALITY_MODES = ("1toN", "Nto1", "NtoM")

# Appended to the papers block in the synthesis (Nto1) mode.
_SYNTHESIS_INSTRUCTION = """

SYNTHESIS REQUIREMENT
---------------------
You were given SEVERAL papers.  Every idea you propose must SYNTHESIZE
mechanisms across at least TWO of them — e.g. one paper's signal conditioned on
another's state variable, or one paper's anomaly measured through another's
lens.  Do not paraphrase a single paper.  List every paper that grounds each
idea in its `source_paper_ids`.
"""


def build_query(data_context: str | None = None,
                focus: str | None = None,
                gaps: Sequence[str] | None = None) -> str:
    """Compose the retrieval query from the run's scope + optional focus/gaps.

    ``data_context`` contributes the field vocabulary (so retrieval favours
    papers whose mechanisms are *computable* here — the same scoping idea the
    brainstorm prompt enforces).  ``focus`` is a free-form steer (e.g. an
    island's theme, P4's community summary); ``gaps`` are under-covered
    mechanisms/categories to weight toward.
    """
    parts: list[str] = [
        "quantitative trading alpha factor cross-sectional return prediction",
    ]
    if focus:
        parts.append(focus)
    if gaps:
        parts.append("underexplored mechanisms: " + ", ".join(gaps))
    if data_context:
        # Field names only — the full prose would drown the signal.
        import re

        fields = re.findall(r"^\s{4}(\w+)\s*:", data_context, flags=re.M)
        if fields:
            parts.append("computable from fields: " + ", ".join(sorted(set(fields))))
    return "\n".join(parts)


def _papers_block(papers: Sequence[RetrievedPaper]) -> str:
    parts = []
    for p in papers:
        parts.append(
            f"=== {p.title} (id={p.paper_id}, published={p.published_date}) ===\n"
            f"{p.text or '(no text available)'}")
    return "\n\n".join(parts) or "(no papers retrieved — rely on your own knowledge)"


_MECHANISM_TAGGING = """

MECHANISM TAGGING
-----------------
The research programme is targeting these under-exploited mechanisms:
{mechanisms}
Prefer ideas that exploit one of them.  For EACH idea, additionally include a
"mechanism" key naming the single mechanism it exploits (one of the above, or
your own short canonical name if the idea genuinely uses a different one).
"""


def retrieve_and_brainstorm(
    llm: Any,
    store: EmbedStore,
    *,
    n_ideas: int,
    known_ids: set[str],
    data_context: str,
    cardinality: str = "1toN",
    k_papers: int = 4,
    cutoff_date: str | None = None,
    focus: str | None = None,
    gaps: Sequence[str] | None = None,
    mechanism_tags: Sequence[str] | None = None,
) -> list[dict]:
    """Retrieve grounding papers and brainstorm ``n_ideas`` raw idea dicts.

    Returns the raw idea dicts (the caller coerces/validates them exactly like
    the oneshot path) with ``source_paper_ids`` already **citation-verified**.
    ``mechanism_tags`` (GraphRAG mode) asks each idea to name the mechanism it
    exploits — the hook that closes the Paper → Mechanism → Factor provenance
    path when the resulting factor is linked back into the graph.
    """
    if cardinality not in CARDINALITY_MODES:
        raise ValueError(f"cardinality must be one of {CARDINALITY_MODES}")

    query = build_query(data_context, focus=focus, gaps=gaps)
    papers = store.retrieve_papers(query, k=k_papers, cutoff_date=cutoff_date)
    retrieved_ids = [p.paper_id for p in papers]
    log.info("retrieved %d paper(s) [%s] for cardinality=%s",
             len(papers), ", ".join(retrieved_ids), cardinality)

    from quant_fund_agent.agents.factor_research.graph import _parse_json
    from quant_fund_agent.agents.factor_research.prompts import BRAINSTORM_PROMPT

    tagging = ""
    if mechanism_tags:
        tagging = _MECHANISM_TAGGING.format(
            mechanisms="\n".join(f"- {m}" for m in mechanism_tags))

    def _call(papers_block: str, n: int) -> list[dict]:
        prompt = BRAINSTORM_PROMPT.format(
            n_ideas=n, data_context=data_context,
            papers_block=papers_block + tagging,
            existing_ids=", ".join(sorted(known_ids)) if known_ids else "(none)")
        resp = llm.invoke(prompt)
        content = getattr(resp, "content", str(resp))
        return _parse_json(content).get("ideas", [])

    raw: list[dict] = []
    if cardinality == "1toN" and papers:
        per_paper = max(1, -(-n_ideas // len(papers)))
        for p in papers:
            if len(raw) >= n_ideas:
                break
            ideas = _call(_papers_block([p]), per_paper)
            for idea in ideas:
                idea["source_paper_ids"] = [p.paper_id]  # authoritative
            raw.extend(ideas)
    elif cardinality == "Nto1":
        raw = _call(_papers_block(papers) + _SYNTHESIS_INSTRUCTION, n_ideas)
    else:  # NtoM (and the no-papers fallback of 1toN)
        raw = _call(_papers_block(papers), n_ideas)

    # ── citation verification ──
    verified_ideas: list[dict] = []
    for idea in raw[: n_ideas * 2]:
        claimed = list(idea.get("source_paper_ids") or [])
        ok, bad = verify_citations(claimed, retrieved_ids)
        if bad:
            log.info("idea %r cited unretrieved paper(s) %s — citations dropped",
                     idea.get("factor_id"), bad)
        idea["source_paper_ids"] = ok
        if cardinality != "1toN" and papers and claimed and not ok:
            # Every citation was hallucinated → the "grounding" is fiction.
            log.info("idea %r dropped: no verifiable grounding", idea.get("factor_id"))
            continue
        verified_ideas.append(idea)
    return verified_ideas[:n_ideas]
