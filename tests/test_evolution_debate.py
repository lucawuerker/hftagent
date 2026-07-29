"""Tests for the P3 agent split: Hypothesis → Debate (skeptic/moderator) → Codegen,
plus per-role model selection."""

from __future__ import annotations

import json

import pytest

from quant_fund_agent.agents.factor_research.debate import (
    book_summary,
    run_debate,
)

IDEA = {
    "factor_id": "vol_conditioned_mom",
    "name": "Vol-conditioned momentum",
    "category": "momentum",
    "trading_idea": "Momentum persists only in low-volatility regimes.",
    "description": "20-bar momentum gated by trailing vol quintile.",
    "prediction_horizon": 6,
    "expected_sign": 1,
    "source_paper_ids": ["p_mom"],
}


class ScriptedDebateLLM:
    """Skeptic/moderator/revision responses scripted per call order."""

    def __init__(self, verdicts, revised_idea=None):
        self.verdicts = list(verdicts)   # one per moderator call
        self.revised_idea = revised_idea or {**IDEA, "trading_idea": "REVISED"}
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)

        class _R:
            content = ""

        if "You are the in-house skeptic" in prompt:
            _R.content = json.dumps({"issues": [
                {"dimension": "redundancy", "severity": "major",
                 "critique": "looks like alpha_007"}], "overall": "meh"})
        elif "You moderate a research debate" in prompt:
            _R.content = json.dumps(
                {"verdict": self.verdicts.pop(0), "guidance": "fix redundancy"})
        elif "requires a revision" in prompt:
            _R.content = json.dumps(self.revised_idea)
        else:
            raise AssertionError(f"unexpected prompt: {prompt[:80]}")
        return _R()


def test_debate_accept_first_round():
    llm = ScriptedDebateLLM(["accept"])
    verdict, final, transcript = run_debate(llm, IDEA, data_context="ctx", book=[])
    assert verdict == "accept"
    assert final == IDEA
    assert len(transcript) == 1 and transcript[0]["verdict"] == "accept"


def test_debate_reject():
    llm = ScriptedDebateLLM(["reject"])
    verdict, final, _ = run_debate(llm, IDEA, data_context="ctx")
    assert verdict == "reject"


def test_debate_revise_then_accept_uses_revised_idea():
    llm = ScriptedDebateLLM(["revise", "accept"])
    verdict, final, transcript = run_debate(llm, IDEA, data_context="ctx")
    assert verdict == "accept"
    assert final["trading_idea"] == "REVISED"
    # skeptic + moderator ran twice, revision once
    assert sum("in-house skeptic" in p for p in llm.prompts) == 2
    assert sum("requires a revision" in p for p in llm.prompts) == 1
    assert any("revised_idea" in t for t in transcript)


def test_debate_unresolved_revise_fails_open():
    """An exhausted revision budget ACCEPTS the latest revision (fail-open):
    the deterministic harness — not the debate — is the arbiter of empirical
    merit, so an unresolved 'revise' must never starve the run."""
    llm = ScriptedDebateLLM(["revise"])
    verdict, _, transcript = run_debate(llm, IDEA, data_context="ctx",
                                        max_revisions=0)
    assert verdict == "accept"
    assert any("budget exhausted" in str(t.get("note", "")) for t in transcript)


def test_debate_fails_open_on_llm_error():
    class Broken:
        def invoke(self, prompt):
            raise RuntimeError("api down")

    verdict, final, transcript = run_debate(Broken(), IDEA, data_context="ctx")
    assert verdict == "accept"
    assert final == IDEA
    assert "error" in transcript[0]


def test_book_summary_renders_and_truncates():
    entries = [{"factor_id": "f1", "category": "momentum",
                "trading_idea": "x" * 500}]
    s = book_summary(entries)
    assert "f1 [momentum]" in s and len(s) < 250
    assert book_summary([]) == "(the book is currently empty)"


# ── per-role model selection ─────────────────────────────────────────────────

def test_role_llm_env_override(monkeypatch):
    import quant_fund_agent.agents.factor_research.evolution.loop as loop_mod

    captured = {}

    def fake_make_chat_llm(model=None, provider=None, **kw):
        captured["model"] = model
        captured["provider"] = provider
        return object()

    monkeypatch.setattr("quant_fund_agent.llm.make_chat_llm", fake_make_chat_llm)
    monkeypatch.setenv("DEBATE_LLM_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("DEBATE_LLM_PROVIDER", "anthropic")

    loop_mod._get_llm(temperature=0.3, role="debate")
    assert captured["model"] == "claude-sonnet-5"
    assert captured["provider"] == "anthropic"

    monkeypatch.delenv("HYPOTHESIS_LLM_MODEL", raising=False)
    loop_mod._get_llm(temperature=0.6, role="hypothesis")
    assert captured["model"] is None  # falls back to the run's research model


# ── loop integration: debate gates children before evaluation ────────────────

class RoleDispatchLLM:
    """One fake serving every role, dispatching on prompt markers."""

    def __init__(self, moderator_verdict: str, child_code_factory):
        self.moderator_verdict = moderator_verdict
        self.child_code_factory = child_code_factory
        self.codegen_calls = 0
        self.hypothesis_calls = 0

    def invoke(self, prompt: str):
        class _R:
            content = ""

        if "propose the HYPOTHESIS" in prompt:
            self.hypothesis_calls += 1
            _R.content = json.dumps({
                "factor_id": "debated_child", "name": "child",
                "category": "momentum", "trading_idea": "smoothed momentum",
                "description": "d", "prediction_horizon": 1,
                "expected_sign": 1, "suggested_horizons": [1]})
        elif "in-house skeptic" in prompt:
            _R.content = json.dumps({"issues": [], "overall": "fine"})
        elif "You moderate a research debate" in prompt:
            _R.content = json.dumps({"verdict": self.moderator_verdict,
                                     "guidance": ""})
        elif "You are writing production Python" in prompt:
            self.codegen_calls += 1
            _R.content = json.dumps({"code": self.child_code_factory()})
        elif "complementary strengths" in prompt:
            # crossover child (single call, code included)
            _R.content = json.dumps({
                "factor_id": "xover_child", "name": "x", "category": "momentum",
                "trading_idea": "t", "description": "d", "prediction_horizon": 1,
                "expected_sign": 1, "code": self.child_code_factory()})
        else:
            raise AssertionError(f"unexpected prompt: {prompt[:80]}")
        return _R()


@pytest.fixture()
def debate_loop(monkeypatch, tmp_path):
    """A wired loop with debate=on over the synthetic panel."""
    from tests.test_evolution_loop import (
        CHILD_BODY_1,
        _factor_code,
        _seeds,
        synthetic_panel,
    )

    panel = synthetic_panel.__wrapped__()  # unwrap the fixture function
    monkeypatch.setenv("QF_USE_MCP", "0")

    from quant_fund_agent.mcp import research_service as svc

    monkeypatch.setattr(svc, "_load_panel_cached",
                        lambda data_dir, fields, n_tickers: panel)
    monkeypatch.setattr(svc, "_panel_cache_key",
                        lambda data_dir, fields, n_tickers: ("test-panel-debate",))
    svc._SIGNAL_CACHE.clear()

    counter = {"n": 0}

    def code_factory():
        counter["n"] += 1
        return _factor_code("debated_child", CHILD_BODY_1)

    import quant_fund_agent.agents.factor_research.evolution.loop as loop_mod
    from quant_fund_agent.agents.factor_research.evolution.loop import (
        EvolutionLoop,
        EvolutionRunConfig,
    )

    def make(verdict: str) -> tuple[EvolutionLoop, RoleDispatchLLM]:
        fake = RoleDispatchLLM(verdict, code_factory)
        monkeypatch.setattr(loop_mod, "_get_llm",
                            lambda temperature, role=None: fake)
        cfg = EvolutionRunConfig(
            generations=1, population_size=6, children_per_generation=3,
            seed=5, target_horizon=1, stability_blocks=4,
            debate="on", p_llm_semantic=1.0, p_crossover=0.0, p_jitter=0.0,
            n_tickers=None, out_dir=str(tmp_path / "evo"))
        loop = EvolutionLoop(cfg, data_context="CTX",
                             fields=["open", "high", "low", "close", "volume"])
        loop._load_known_ids = lambda: None
        return loop, fake

    return make, _seeds


def test_debate_on_accept_runs_hypothesis_then_codegen(debate_loop):
    make, seeds = debate_loop
    loop, fake = make("accept")
    summary = loop.run(initial_programs=seeds())
    assert fake.hypothesis_calls > 0
    assert fake.codegen_calls > 0
    # at least one debated child was admitted (billed) beyond the two seeds
    assert summary["n_trials"] > 2


def test_debate_on_reject_blocks_codegen_and_billing(debate_loop):
    make, seeds = debate_loop
    loop, fake = make("reject")
    summary = loop.run(initial_programs=seeds())
    assert fake.hypothesis_calls > 0
    assert fake.codegen_calls == 0            # rejected pre-codegen
    assert summary["n_trials"] == 2           # only the seeds were evaluated
