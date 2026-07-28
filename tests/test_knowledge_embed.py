"""Tests for the RAG subsystem (P2): embed store, retrieval, cardinality modes,
date-gating, data-scope gating and citation verification.  All offline via the
hashing embedder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from quant_fund_agent.knowledge.embed_store import (
    CorpusDoc,
    EmbedStore,
    hashing_embedder,
    verify_citations,
)
from quant_fund_agent.knowledge.retrieval import (
    _papers_block,
    build_query,
    retrieve_and_brainstorm,
)
from quant_fund_agent.knowledge.embed_store import RetrievedPaper


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


# ── data-scope gating ────────────────────────────────────────────────────────

def _scoped_corpus() -> list[CorpusDoc]:
    return [
        CorpusDoc("s_price", "Momentum on prices", "2015-03-01",
                  "Momentum momentum winners minus losers price trend "
                  "continuation daily bars." * 3, data_scope="price"),
        CorpusDoc("s_fund", "Accruals anomaly", "2016-05-01",
                  "Accruals anomaly earnings quality fundamentals balance "
                  "sheet profitability cross-section." * 3,
                  data_scope="fundamental"),
        CorpusDoc("s_gen", "Hawkes processes", "2019-02-01",
                  "Hawkes process self-exciting point process intensity "
                  "kernel estimation." * 3, data_scope="general"),
        CorpusDoc("s_legacy", "Untagged legacy paper", "2014-01-01",
                  "Momentum accruals Hawkes a bit of everything, indexed "
                  "before scope tagging existed." * 3),  # data_scope=None
    ]


@pytest.fixture()
def scoped_store(tmp_path):
    return EmbedStore(_scoped_corpus(), embedder="hash",
                      cache_dir=tmp_path / "emb_scoped").build()


def test_scope_mask_filters_tagged_docs_and_passes_untagged(scoped_store):
    got = scoped_store.retrieve_papers(
        "anything at all", k=10, allowed_scopes={"price"})
    ids = {p.paper_id for p in got}
    assert "s_price" in ids
    assert "s_legacy" in ids            # untagged always passes
    assert "s_fund" not in ids
    assert "s_gen" not in ids

    got = scoped_store.retrieve_papers(
        "anything at all", k=10, allowed_scopes={"price", "fundamental"})
    ids = {p.paper_id for p in got}
    assert ids == {"s_price", "s_fund", "s_legacy"}

    # no filter → everything retrievable
    got = scoped_store.retrieve_papers("anything at all", k=10)
    assert len(got) == 4


def test_scope_mask_composes_with_cutoff_date(scoped_store):
    got = scoped_store.retrieve_papers(
        "anything", k=10, cutoff_date="2016-01-01",
        allowed_scopes={"fundamental"})
    ids = {p.paper_id for p in got}
    # s_fund is scope-allowed but published after the cutoff; s_legacy is
    # untagged (passes scope) and pre-cutoff.
    assert ids == {"s_legacy"}


def test_chunks_inherit_parent_doc_scope(scoped_store):
    chunks = scoped_store.retrieve_chunks(
        "Hawkes self-exciting intensity", k=10,
        allowed_scopes={"fundamental"})
    pids = {c.paper_id for c in chunks}
    assert "s_gen" not in pids          # its chunks are masked with the doc
    assert "s_price" not in pids
    assert pids <= {"s_fund", "s_legacy"}

    chunks = scoped_store.retrieve_chunks(
        "Hawkes self-exciting intensity", k=3, allowed_scopes={"general"})
    assert chunks and chunks[0].paper_id == "s_gen"


def test_data_scope_persisted_in_meta_json(scoped_store, tmp_path):
    meta = json.loads((tmp_path / "emb_scoped" / "meta.json").read_text())
    scopes = dict(zip(meta["paper_ids"], meta["data_scopes"]))
    assert scopes == {"s_price": "price", "s_fund": "fundamental",
                      "s_gen": "general", "s_legacy": None}


def test_retrieve_and_brainstorm_forwards_allowed_scopes(scoped_store):
    llm = FakeBrainstormLLM()
    retrieve_and_brainstorm(
        llm, scoped_store, n_ideas=2, known_ids=set(), data_context="ctx",
        cardinality="NtoM", k_papers=4, allowed_scopes={"general"})
    prompt = llm.prompts[0]
    assert "id=s_gen," in prompt
    assert "id=s_legacy," in prompt     # untagged passes
    assert "id=s_fund," not in prompt
    assert "id=s_price," not in prompt


# ── paper-text token cap ─────────────────────────────────────────────────────

def test_papers_block_truncates_long_text(monkeypatch):
    monkeypatch.setenv("QF_RAG_PAPER_MAX_CHARS", "100")
    long = RetrievedPaper("p_long", "Long paper", "2020-01-01", 0.9,
                          text="x" * 500)
    short = RetrievedPaper("p_short", "Short paper", "2020-01-01", 0.8,
                           text="y" * 50)
    block = _papers_block([long, short])
    assert "... [truncated]" in block
    assert "x" * 100 in block and "x" * 101 not in block
    assert "y" * 50 in block            # under the cap → untouched
    # the truncated suffix is attached to the long paper only
    assert block.count("[truncated]") == 1


def test_papers_block_default_cap_is_20k(monkeypatch):
    monkeypatch.delenv("QF_RAG_PAPER_MAX_CHARS", raising=False)
    p = RetrievedPaper("p", "T", "2020-01-01", 0.5, text="z" * 25_000)
    block = _papers_block([p])
    assert "... [truncated]" in block
    assert "z" * 20_000 in block and "z" * 20_001 not in block


def test_papers_block_cap_applies_inside_brainstorm(scoped_store, monkeypatch):
    monkeypatch.setenv("QF_RAG_PAPER_MAX_CHARS", "80")
    llm = FakeBrainstormLLM()
    retrieve_and_brainstorm(
        llm, scoped_store, n_ideas=2, known_ids=set(), data_context="ctx",
        cardinality="Nto1", k_papers=3)
    assert "[truncated]" in llm.prompts[0]


# ── populate_papers block structure ──────────────────────────────────────────

def _load_populate_papers():
    path = Path(__file__).resolve().parent.parent / "scripts" / "populate_papers.py"
    spec = importlib.util.spec_from_file_location("populate_papers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pp():
    return _load_populate_papers()


def test_query_blocks_expose_scope_labels(pp):
    assert set(pp.QUERY_BLOCKS) == {"price", "fundamental", "general"}
    for name, block in pp.QUERY_BLOCKS.items():
        assert len(block["queries"]) >= 15 or name == "price"
        assert block["categories"]
        assert block["max_results_per_query"] > 0
    # fundamental keeps the q-fin/econ restriction and adds q-fin.GN
    fund_cats = set(pp.QUERY_BLOCKS["fundamental"]["categories"])
    assert {"q-fin.GN", "q-fin.PM", "q-fin.ST", "econ.GN"} <= fund_cats
    # general has NO q-fin restriction — pure maths/ML/signal-processing
    gen_cats = set(pp.QUERY_BLOCKS["general"]["categories"])
    assert not any(c.startswith("q-fin") for c in gen_cats)
    assert {"math.PR", "math.ST", "stat.ML", "cs.LG", "eess.SP",
            "math.DS"} <= gen_cats
    # tuned toward ~+1000 new papers across the two new blocks
    expected_new = sum(
        len(pp.QUERY_BLOCKS[b]["queries"]) *
        pp.QUERY_BLOCKS[b]["max_results_per_query"]
        for b in ("fundamental", "general"))
    assert 900 <= expected_new <= 1500


def test_collect_candidates_stamps_scope_and_dedups(pp):
    blocks = {
        "fundamental": {"queries": ["q1"], "categories": ["q-fin.GN"],
                        "max_results_per_query": 5},
        "general": {"queries": ["q2"], "categories": ["math.PR"],
                    "max_results_per_query": 5},
    }
    payload = {
        "q1": [{"arxiv_id": "1111.1", "title": "A", "authors": ["x"],
                "abstract": "a", "published_date": None,
                "url": "u1", "pdf_url": "p1"},
               {"arxiv_id": "2222.2", "title": "B", "authors": ["x"],
                "abstract": "b", "published_date": None,
                "url": "u2", "pdf_url": "p2"}],
        "q2": [{"arxiv_id": "2222.2", "title": "B", "authors": ["x"],
                "abstract": "b", "published_date": None,
                "url": "u2", "pdf_url": "p2"},   # dup across blocks
               {"arxiv_id": "3333.3", "title": "C", "authors": ["x"],
                "abstract": "c", "published_date": None,
                "url": "u3", "pdf_url": "p3"}],
    }
    calls = []

    def fake_fetch(query, max_results, categories):
        calls.append((query, max_results, tuple(categories)))
        return payload[query]

    seen = {"3333.3"}  # already in the DB → skipped
    cands = pp.collect_candidates(blocks, seen, max_candidates=100,
                                  fetch=fake_fetch, sleep=lambda s: None)
    assert calls == [("q1", 5, ("q-fin.GN",)), ("q2", 5, ("math.PR",))]
    got = {c["arxiv_id"]: c["data_scope"] for c in cands}
    assert got == {"1111.1": "fundamental", "2222.2": "fundamental"}


def test_build_paper_stamps_data_scope_metadata(pp):
    cand = {"arxiv_id": "1234.5", "title": "Accruals & Quality!",
            "authors": ["A. Uthor"], "abstract": "abs",
            "published_date": None, "url": "http://x",
            "pdf_url": "http://x.pdf", "data_scope": "fundamental"}
    existing = {"accruals_quality"}
    paper = pp.build_paper(cand, "desc", existing)
    assert paper.metadata["data_scope"] == "fundamental"
    assert paper.metadata["arxiv_id"] == "1234.5"
    assert paper.metadata["description"] == "desc"
    assert paper.id != "accruals_quality" and paper.id in existing

    # a candidate without a scope (defensive) → None, i.e. legacy semantics
    cand2 = dict(cand, arxiv_id="9999.9", title="Other title")
    paper2 = pp.build_paper(cand2 | {"data_scope": None}, "d", existing)
    assert paper2.metadata["data_scope"] is None
