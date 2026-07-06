"""Tests for the cross-run experience memory (WS5).

The memory records survivors AND controlled negative evidence (per-mechanism attempt
tallies) so exhaustion is detectable, is keyed per-config, and round-trips on the
KnowledgeGraph file.  All offline — no market data.
"""

from __future__ import annotations

from quant_fund_agent.knowledge import experience
from quant_fund_agent.knowledge.graph_store import KnowledgeGraph


def test_attempt_tally_records_survivors_and_failures():
    g = KnowledgeGraph()
    # a mechanism tried 6× that never survives → exhausted
    for i in range(6):
        experience.record_attempt(g, "crowded_momentum", survived=False,
                                  marginal=-0.001, generation=i)
    # a mechanism tried 4× that mostly survives → healthy
    for i in range(4):
        experience.record_attempt(g, "vol_state", survived=True,
                                  marginal=0.02, generation=i)
    mid = KnowledgeGraph.mechanism_id("crowded_momentum")
    assert g.g.nodes[mid]["n_attempts"] == 6
    assert g.g.nodes[mid].get("n_survived", 0) == 0
    assert g.g.nodes[mid]["last_generation"] == 5


def test_exhausted_detection_needs_negative_evidence():
    """A tried-and-failed mechanism must read as EXHAUSTED, not unexplored — which
    only works because we record attempts, not just survivors."""
    g = KnowledgeGraph()
    for i in range(8):
        experience.record_attempt(g, "stale_reversal", survived=False, generation=i)
    for i in range(8):
        experience.record_attempt(g, "good_carry", survived=(i % 2 == 0), generation=i)
    exhausted = experience.exhausted_mechanisms(g, min_attempts=5,
                                                max_survival_rate=0.25)
    assert "stale_reversal" in exhausted     # 0/8 survived → exhausted
    assert "good_carry" not in exhausted      # 4/8 survived → healthy
    # a barely-tried mechanism is NOT exhausted (too few attempts to judge)
    experience.record_attempt(g, "fresh_idea", survived=False)
    assert "fresh_idea" not in experience.exhausted_mechanisms(g, min_attempts=5)


def test_record_survivor_stamps_performance_and_provenance():
    g = KnowledgeGraph()
    experience.record_survivor(g, "alpha_1", val_ic=0.03, marginal_value=0.012,
                               generation=3, mechanism_name="vol_state",
                               objective={"marginal_value": 0.012})
    fnode = KnowledgeGraph.factor_id("alpha_1")
    assert g.g.nodes[fnode]["realized_marginal_value"] == 0.012
    assert g.g.nodes[fnode]["generation"] == 3
    # provenance edge mechanism —realized_by→ factor
    assert fnode in g.factors_for_mechanism(KnowledgeGraph.mechanism_id("vol_state"))


def test_memory_summary_is_empty_until_something_is_tried():
    g = KnowledgeGraph()
    assert experience.memory_summary(g) == ""
    experience.record_attempt(g, "m", survived=True, marginal=0.01)
    assert "EXPERIENCE MEMORY" in experience.memory_summary(g)


def test_memory_is_idempotent_on_reload(tmp_path):
    """Across-run accrual: save, reload, keep incrementing — counters accumulate,
    they do not reset or double-count within a run."""
    path = tmp_path / "cfg" / "graph.json"
    g = KnowledgeGraph()
    experience.record_attempt(g, "m", survived=True, generation=0)
    g.save(path)
    # a fresh run loads the memory and records one more attempt
    g2 = experience.load_or_new(path)
    experience.record_attempt(g2, "m", survived=False, generation=1)
    mid = KnowledgeGraph.mechanism_id("m")
    assert g2.g.nodes[mid]["n_attempts"] == 2       # accrued across the two runs
    assert g2.g.nodes[mid]["n_survived"] == 1


def test_memory_path_is_per_config():
    p_sp = experience.memory_graph_path("yfinance_equity_sp100")
    p_lob = experience.memory_graph_path("lobster_etf")
    assert p_sp != p_lob
    assert p_sp.name == "graph.json"


def test_load_or_new_tolerates_missing_and_corrupt(tmp_path):
    assert isinstance(experience.load_or_new(tmp_path / "nope.json"), KnowledgeGraph)
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json")
    assert isinstance(experience.load_or_new(bad), KnowledgeGraph)  # degrades, no crash
