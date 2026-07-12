"""Tests for E3: the exec skeptic debate (fail-open) + RAG splice."""

from __future__ import annotations

import json

import pytest

from quant_fund_agent.agents.execution_research.evolution.debate import (
    build_skeptic_prompt,
    execution_literature_snippets,
    run_exec_debate,
)

PAYLOAD = {"executor_id": "deb_child", "code": "x = 1",
           "mechanism": "vol scaling", "expected_effect": "less drawdown",
           "regime": "per_underlying"}


class _LLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        self.calls += 1
        return self.responses.pop(0)


def test_skeptic_prompt_carries_attack_lines_and_archive():
    p = build_skeptic_prompt(PAYLOAD, ["momentum decay exit", "turnover budget"])
    assert "COST REALISM" in p and "BACKTEST ARTIFACTS" in p
    assert "momentum decay exit" in p
    assert PAYLOAD["code"] in p


def test_debate_accept_passes_through():
    llm = _LLM(['{"verdict": "accept", "critique": "fine"}'])
    verdict, final, transcript = run_exec_debate(llm, PAYLOAD)
    assert verdict == "accept"
    assert final["code"] == "x = 1"
    assert transcript[0]["verdict"] == "accept"


def test_debate_reject_drops_the_child():
    llm = _LLM(['{"verdict": "reject", "critique": "leverage disguise"}'])
    verdict, _, transcript = run_exec_debate(llm, PAYLOAD)
    assert verdict == "reject"
    assert "leverage" in transcript[0]["critique"]


def test_debate_revise_then_accept_uses_revised_code():
    llm = _LLM([
        '{"verdict": "revise", "critique": "too churny", '
        '"required_change": "add hysteresis"}',
        json.dumps({"executor_id": "ignored", "code": "x = 2",
                    "mechanism": "vol scaling + hysteresis"}),
        '{"verdict": "accept", "critique": "better"}',
    ])
    verdict, final, transcript = run_exec_debate(llm, PAYLOAD)
    assert verdict == "accept"
    assert final["code"] == "x = 2"
    assert final["executor_id"] == "deb_child"      # id survives the revision
    assert len(transcript) == 2


def test_debate_unresolved_revise_becomes_reject():
    llm = _LLM([
        '{"verdict": "revise", "critique": "a", "required_change": "b"}',
        json.dumps({"executor_id": "deb_child", "code": "x = 3"}),
        '{"verdict": "revise", "critique": "still bad", "required_change": "c"}',
    ])
    verdict, _, _ = run_exec_debate(llm, PAYLOAD)
    assert verdict == "reject"


def test_debate_fails_open_on_garbage():
    llm = _LLM(["the skeptic mumbles incoherently"])
    verdict, final, transcript = run_exec_debate(llm, PAYLOAD)
    assert verdict == "accept"
    assert "error" in transcript[0]


def test_rag_snippets_never_raise(monkeypatch):
    # whatever the corpus state, the splice must return a string
    out = execution_literature_snippets(k=2)
    assert isinstance(out, str)


def test_loop_debate_rejects_before_evaluation(tmp_path, monkeypatch):
    """A debate-rejected child never reaches the harness (no n_trials billed)."""
    import numpy as np
    import pandas as pd

    monkeypatch.setenv("QF_USE_MCP", "0")
    from quant_fund_agent.mcp import research_service as svc

    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-01-01", periods=480, freq="D")
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (480, 6)), axis=0),
        index=idx, columns=list("ABCDEF"))
    monkeypatch.setattr(svc, "_load_panel_cached",
                        lambda data_dir, fields, n_tickers: {"close": close})
    monkeypatch.setattr(svc, "_panel_cache_key",
                        lambda data_dir, fields, n_tickers: ("deb-panel",))
    svc._SIGNAL_CACHE.clear()

    factor_code = (
        '"""f"""\nfrom __future__ import annotations\n\nimport pandas as pd\n\n'
        "from quant_fund_agent.factors.base import BaseFactor\n"
        "from quant_fund_agent.factors.registry import register_factor\n\n\n"
        "@register_factor\nclass FDebMom(BaseFactor):\n"
        '    factor_id = "deb_mom"\n    name = "deb_mom"\n'
        '    category = "momentum"\n    inputs = ["close"]\n'
        "    prediction_horizon = 1\n\n"
        "    def calc(self, data):\n        close = data[\"close\"]\n"
        "        return close.pct_change().fillna(0.0)\n")
    frozen = svc.freeze_signals([{"factor_id": "deb_mom", "code": factor_code}],
                                out_dir=str(tmp_path), version=1,
                                target_horizon=1,
                                specs=[{"model": "ridge", "subset": [0]}])
    assert frozen["ok"], frozen.get("error")

    from quant_fund_agent.agents.execution_research.evolution.loop import (
        ExecEvolutionLoop,
        ExecEvolutionRunConfig,
    )
    import quant_fund_agent.agents.execution_research.evolution.loop as loop_mod

    class RejectingLLM:
        def invoke(self, prompt):
            if "SKEPTIC" in prompt:
                return '{"verdict": "reject", "critique": "artifact exploitation"}'
            return json.dumps({"executor_id": "whatever", "code": "x = 1",
                               "regime": "per_underlying"})

    monkeypatch.setattr(loop_mod, "_get_llm",
                        lambda temperature, role=None: RejectingLLM())

    loop = ExecEvolutionLoop(ExecEvolutionRunConfig(
        out_dir=str(tmp_path / "evo"), signals_manifest=frozen["manifest_path"],
        generations=1, population_size=4, children_per_generation=2,
        seed=1, n_tickers=None,
        p_llm_semantic=1.0, p_crossover=0.0, p_jitter=0.0, debate="on"))
    loop.run()

    seeds_billed = 2                                # only the 2 seeds scored
    assert loop.controller.n_trials == seeds_billed
    assert loop.debate_transcripts
    assert all(t["verdict"] == "reject" for t in loop.debate_transcripts)
