"""Knowledge layer for the evolutionary factor researcher (RAG + GraphRAG).

* :mod:`~quant_fund_agent.knowledge.embed_store` — whole-paper + chunk
  embeddings over the paper library, cosine retrieval with cutoff-date gating,
  and citation verification (P2).
* :mod:`~quant_fund_agent.knowledge.retrieval` — retrieval-grounded idea
  generation: query construction and the cardinality modes ``1→N`` / ``N→1`` /
  ``N→M`` (P2), consumed by the evolution loop's seeding and (P3) the Prompter.
* :mod:`~quant_fund_agent.knowledge.graph_store` / ``graph_build`` /
  ``graph_query`` / ``empirical_edges`` — the custom hybrid GraphRAG (P4).
"""
