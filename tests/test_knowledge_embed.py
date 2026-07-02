"""Tests for the RAG subsystem (P2): embed store, retrieval, cardinality modes,
date-gating and citation verification.  All offline via the hashing embedder."""

from __future__ import annotations

import json

import numpy as np
import pytest

from quant_fund_agent.knowledge.embed_store import (
    CorpusDoc,
    EmbedStore,
    hashing_embedder,
    verify_citations,
)
from quant_fund_agent.knowledge.retrieval import (
    build_query,
    retrieve_and_brainstorm,
)


def _corpus() -> list[CorpusDoc]:
    return [
        CorpusDoc("p_mom", "Momentum returns", "2015-03-01",
                  "Momentum momentum momentum winners minus losers persists "
                  "cross-sectional stock returns trend continuation." * 3),
        CorpusDoc("p_vol", "Volatility anomaly", "2018-06-01",
                  "Idiosyncratic volatility low vol anomaly risk lottery "
                  "preference variance premium." * 3),
        CorpusDoc("p_liq", "Liquidity provision", "2021-01-01",
                  "Liquidity provision bid ask spread order flow imbalance "
                  "market making inventory microstructure." * 3),
        CorpusDoc("p_undated", "Mystery paper", None,
                  "Momentum with unknown publication date." * 3),
    ]


@pytest.fixture()
def store(tmp_path):
    return EmbedStore(_corpus(), embedder="hash", cache_dir=tmp_path / "emb").build()


# ── embedders ────────────────────────────────────────────────────────────────

def test_hashing_embedder_is_deterministic_and_normalised():
    v1 = hashing_embedder(["momentum returns", "volatility"])
    v2 = hashing_embedder(["momentum returns", "volatility"])
    assert np.allclose(v1, v2)
    assert np.allclose(np.linalg.norm(v1, axis=1), 1.0)


# ── retrieval ────────────────────────────────────────────────────────────────

def test_retrieval_ranks_topically(store):
    top = store.retrieve_papers("momentum winners trend", k=2)
    assert top[0].paper_id in ("p_mom", "p_undated")
    ids = {p.paper_id for p in top}
    assert "p_vol" not in ids

    vol = store.retrieve_papers("volatility anomaly lottery", k=1)
    assert vol[0].paper_id == "p_vol"


def test_cutoff_date_gating_excludes_late_and_undated(store):
    got = store.retrieve_papers("liquidity order flow", k=4, cutoff_date="2019-01-01")
    ids = {p.paper_id for p in got}
    assert "p_liq" not in ids        # published 2021 > cutoff
    assert "p_undated" not in ids    # unprovable → excluded
    assert ids <= {"p_mom", "p_vol"}


def test_chunk_retrieval_returns_grounding_text(store):
    chunks = store.retrieve_chunks("bid ask spread market making", k=3)
    assert chunks and chunks[0].paper_id == "p_liq"
    assert "spread" in chunks[0].text.lower()


def test_cache_roundtrip_skips_reembedding(tmp_path):
    calls = {"n": 0}

    def counting_embedder(texts):
        calls["n"] += 1
        return hashing_embedder(texts)

    s1 = EmbedStore(_corpus(), embedder="hash", cache_dir=tmp_path / "emb")
    s1.embed = counting_embedder
    s1.build()
    built_calls = calls["n"]
    assert built_calls >= 2  # papers + chunks

    s2 = EmbedStore(_corpus(), embedder="hash", cache_dir=tmp_path / "emb")
    s2.embed = counting_embedder
    s2.build()
    assert calls["n"] == built_calls           # cache hit — no re-embedding
    assert s2.paper_vecs.shape == s1.paper_vecs.shape

    # corpus change invalidates the cache
    docs = _corpus()
    docs[0].text += " extra sentence."
    s3 = EmbedStore(docs, embedder="hash", cache_dir=tmp_path / "emb")
    s3.embed = counting_embedder
    s3.build()
    assert calls["n"] > built_calls


# ── citation verification ────────────────────────────────────────────────────

def test_verify_citations_splits_hallucinations():
    ok, bad = verify_citations(["p_mom", "p_fake"], ["p_mom", "p_vol"])
    assert ok == ["p_mom"] and bad == ["p_fake"]


# ── cardinality modes ─────────────────────────────────────────────────────────

class FakeBrainstormLLM:
    """Returns ideas citing the papers named in the prompt + one hallucination."""

    def __init__(self):
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        import re

        ids = re.findall(r"id=(\w+),", prompt)
        ideas = [{
            "factor_id": f"idea_{len(self.prompts)}_{k}",
            "name": "x", "category": "momentum", "trading_idea": "t",
            "description": "d", "prediction_horizon": 6,
            "source_paper_ids": (ids[:2] if ids else []) + ["p_hallucinated"],
        } for k in range(2)]

        class _R:
            content = json.dumps({"ideas": ideas})

        return _R()


def test_cardinality_1toN_one_call_per_paper_authoritative_citations(store):
    llm = FakeBrainstormLLM()
    ideas = retrieve_and_brainstorm(
        llm, store, n_ideas=4, known_ids=set(), data_context="ctx",
        cardinality="1toN", k_papers=2)
    assert len(llm.prompts) == 2                      # one call per paper
    assert len(ideas) == 4
    for idea in ideas:
        assert len(idea["source_paper_ids"]) == 1     # attributed to its own paper
        assert idea["source_paper_ids"][0] != "p_hallucinated"


def test_cardinality_Nto1_synthesis_call_and_citation_filter(store):
    llm = FakeBrainstormLLM()
    ideas = retrieve_and_brainstorm(
        llm, store, n_ideas=2, known_ids=set(), data_context="ctx",
        cardinality="Nto1", k_papers=3)
    assert len(llm.prompts) == 1                      # ONE synthesis call
    assert "SYNTHESIS REQUIREMENT" in llm.prompts[0]
    assert ideas
    for idea in ideas:
        assert "p_hallucinated" not in idea["source_paper_ids"]
        assert idea["source_paper_ids"]               # verified grounding kept


def test_cardinality_NtoM_single_call(store):
    llm = FakeBrainstormLLM()
    ideas = retrieve_and_brainstorm(
        llm, store, n_ideas=2, known_ids=set(), data_context="ctx",
        cardinality="NtoM", k_papers=3)
    assert len(llm.prompts) == 1
    assert "SYNTHESIS REQUIREMENT" not in llm.prompts[0]
    assert len(ideas) == 2


def test_build_query_includes_scope_fields():
    ctx = "    close     : end-of-bar price\n    volume    : traded volume\n"
    q = build_query(ctx, focus="microstructure island", gaps=["carry"])
    assert "close" in q and "volume" in q
    assert "microstructure island" in q and "carry" in q
