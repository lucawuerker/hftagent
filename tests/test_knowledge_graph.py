"""Tests for the hybrid GraphRAG subsystem (P4): store schema/persistence,
LLM extraction build (fake LLM), communities, empirical edges, gap-finding and
local grounding queries."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.knowledge.embed_store import CorpusDoc
from quant_fund_agent.knowledge.empirical_edges import (
    link_factor_to_mechanism,
    refresh_factor_correlations,
    refresh_field_usage,
    refresh_mechanism_coverage,
)
from quant_fund_agent.knowledge.graph_build import (
    build_graph,
    detect_and_summarize_communities,
    ingest_paper,
)
from quant_fund_agent.knowledge.graph_query import (
    computable_unexploited,
    island_focus,
    local_context,
    mechanism_gaps,
    under_covered_communities,
)
from quant_fund_agent.knowledge.graph_store import KnowledgeGraph, slugify

FIELDS = ["open", "high", "low", "close", "volume", "orderFlow"]


class FakeExtractionLLM:
    """Two-mechanism extraction for momentum papers, one for micro papers."""

    def invoke(self, prompt: str):
        class _R:
            content = ""

        if "id=mom" in prompt:
            payload = {"mechanisms": [
                {"name": "Momentum Spillovers", "description": "winners lift peers",
                 "required_fields": ["close", "volume"],
                 "anomaly": "momentum", "related_mechanisms": ["underreaction"]},
                {"name": "underreaction", "description": "slow news diffusion",
                 "required_fields": ["close"], "anomaly": None,
                 "related_mechanisms": []},
            ]}
        else:
            payload = {"mechanisms": [
                {"name": "order flow pressure", "description": "flow moves price",
                 "required_fields": ["orderFlow", "notAField"],
                 "anomaly": None, "related_mechanisms": []},
            ]}
        _R.content = json.dumps(payload)
        return _R()


def _docs():
    return [
        CorpusDoc("mom1", "Momentum spillover paper", "2016-01-01", "mom text"),
        CorpusDoc("mom2", "More momentum", "2017-01-01", "mom text 2"),
        CorpusDoc("micro1", "Order flow paper", "2019-01-01", "micro text"),
    ]


@pytest.fixture()
def built_graph() -> KnowledgeGraph:
    return build_graph(FakeExtractionLLM(), _docs(), field_vocab=FIELDS)


# ── store + build ────────────────────────────────────────────────────────────

def test_slugify_canonicalises_case_and_plurals():
    assert slugify("Momentum Spillovers") == slugify("momentum spillover")


def test_build_merges_mechanisms_and_gates_fields(built_graph):
    g = built_graph
    # both momentum papers propose the SAME canonical mechanism node
    mid = g.mechanism_id("Momentum Spillovers")
    assert set(g.papers_for_mechanism(mid)) == {"paper:mom1", "paper:mom2"}
    # hallucinated field never entered the graph
    assert g.field_id("notAField") not in g.g
    assert g.field_id("orderFlow") in g.g
    # related_to edge exists between sibling mechanisms
    assert g.mechanism_id("underreaction") in \
        g.neighbors_by_relation(mid, "related_to")
    counts = g.summary()
    assert counts["paper"] == 3 and counts["mechanism"] == 3
    assert counts["anomaly"] == 1


def test_build_is_idempotent(built_graph):
    n_edges = built_graph.g.number_of_edges()
    again = build_graph(FakeExtractionLLM(), _docs(), graph=built_graph,
                        field_vocab=FIELDS)
    assert again.g.number_of_edges() == n_edges  # nothing re-ingested


def test_graph_roundtrip(tmp_path, built_graph):
    p = tmp_path / "graph.json"
    built_graph.save(p)
    loaded = KnowledgeGraph.load(p)
    assert loaded.summary() == built_graph.summary()
    assert set(loaded.g.edges()) == set(built_graph.g.edges())


def test_communities_and_summaries(built_graph):
    summaries = detect_and_summarize_communities(built_graph, seed=0)
    assert summaries
    # momentum mechanisms cluster away from the microstructure one
    mom = built_graph.g.nodes[built_graph.mechanism_id("Momentum Spillovers")]
    micro = built_graph.g.nodes[built_graph.mechanism_id("order flow pressure")]
    assert mom["community"] != micro["community"]
    assert built_graph.g.graph["community_summaries"]


# ── empirical edges ──────────────────────────────────────────────────────────

def test_refresh_factor_correlations_writes_thresholded_edges(built_graph):
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    rng = np.random.default_rng(0)
    base = pd.DataFrame(rng.standard_normal((200, 4)), index=idx,
                        columns=list("ABCD"))
    signals = {
        "f_a": base,
        "f_b": base * 0.9 + 0.1 * rng.standard_normal((200, 4)),  # ~corr 1
        "f_c": pd.DataFrame(rng.standard_normal((200, 4)), index=idx,
                            columns=list("ABCD")),                 # ~corr 0
    }
    n = refresh_factor_correlations(built_graph, signals, threshold=0.5)
    assert n == 1
    edge = built_graph.g.edges[built_graph.factor_id("f_a"),
                               built_graph.factor_id("f_b")]
    assert edge["relation"] == "corr" and edge["weight"] > 0.9
    # refresh replaces, never accretes
    n2 = refresh_factor_correlations(built_graph, signals, threshold=0.5)
    assert n2 == 1
    corr_edges = [e for *e, d in built_graph.g.edges(data=True)
                  if d.get("relation") == "corr"]
    assert len(corr_edges) == 1


def test_field_usage_and_coverage(built_graph):
    refresh_field_usage(built_graph, {"f_mom": ["close", "volume"]})
    assert built_graph.field_id("close") in built_graph.neighbors_by_relation(
        built_graph.factor_id("f_mom"), "uses")

    link_factor_to_mechanism(built_graph, "f_mom", "Momentum Spillovers")
    refresh_mechanism_coverage(built_graph, {"f_mom": 0.04})
    mech = built_graph.g.nodes[built_graph.mechanism_id("Momentum Spillovers")]
    assert mech["n_factors"] == 1
    assert mech["mean_abs_ic"] == pytest.approx(0.04)


# ── queries ──────────────────────────────────────────────────────────────────

def test_mechanism_gaps_and_computability(built_graph):
    # nothing realised yet → every mechanism is a gap, best-documented first
    gaps = mechanism_gaps(built_graph)
    assert gaps[0] == built_graph.mechanism_id("Momentum Spillovers")

    # orderFlow out of scope → the micro mechanism is not computable
    comp = computable_unexploited(built_graph, ["close", "volume"])
    assert built_graph.mechanism_id("order flow pressure") not in comp
    assert built_graph.mechanism_id("Momentum Spillovers") in comp

    # realising a factor removes the gap
    link_factor_to_mechanism(built_graph, "f_mom", "Momentum Spillovers")
    assert built_graph.mechanism_id("Momentum Spillovers") not in \
        mechanism_gaps(built_graph)


def test_under_covered_communities_rank_thinnest_first(built_graph):
    detect_and_summarize_communities(built_graph, seed=0)
    link_factor_to_mechanism(built_graph, "f_mom", "Momentum Spillovers")
    ranked = under_covered_communities(built_graph, k=3)
    assert ranked
    micro_comm = built_graph.g.nodes[
        built_graph.mechanism_id("order flow pressure")]["community"]
    assert ranked[0][0] == micro_comm         # zero factors → thinnest
    assert ranked[0][1] == 0.0


def test_local_context_bundles_papers_and_relations(built_graph):
    ctx = local_context(built_graph, "Momentum Spillovers")
    assert {p["paper_id"] for p in ctx["papers"]} == {"mom1", "mom2"}
    assert "underreaction" in ctx["related_mechanisms"]
    assert set(ctx["needed_fields"]) == {"close", "volume"}
    assert local_context(built_graph, "no such mechanism") == {}


def test_island_focus_steers_by_community_gaps(built_graph):
    detect_and_summarize_communities(built_graph, seed=0)
    focuses = island_focus(built_graph, 2, FIELDS)
    assert len(focuses) == 2
    assert any("target mechanisms" in f for f in focuses)
    joined = " ".join(focuses).lower()
    assert "momentum" in joined or "order flow" in joined
