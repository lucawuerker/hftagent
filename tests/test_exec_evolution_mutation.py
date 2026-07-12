"""Tests for E2: exec LLM-semantic mutation prompts, parsing, reflection briefs.

The LLM is a deterministic fake — these tests prove the *plumbing*: prompt
determinism, robust parsing, validation-before-evaluation, and that the
deterministic reflection brief renders the right advice for each failure mode.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.agents.execution_research.evolution.loop import (
    ExecEvolutionLoop,
    ExecEvolutionRunConfig,
)
from quant_fund_agent.agents.execution_research.evolution.mutation import (
    build_exec_crossover_prompt,
    build_exec_mutation_prompt,
    parse_exec_child_response,
)
from quant_fund_agent.agents.execution_research.evolution.reflection import (
    exec_mutation_brief,
)
from quant_fund_agent.agents.execution_research.evolution.seeds import (
    seed_execution_programs,
)
from quant_fund_agent.research_eval.fitness import (
    FitnessResult,
    GateResults,
    ObjectiveVector,
)

# a valid child module the fake LLM will emit
CHILD_CODE = '''\
"""Vol-scaled sign executor (test child)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.execution.base import BaseExecutor, register_executor


@register_executor
class VolScaledSign(BaseExecutor):
    executor_id = "CHILD_ID"
    name = "vol-scaled sign"
    regime = "per_underlying"
    inputs = ["signal", "vol"]
    params = {"n_max": 6, "vol_floor": 0.005}

    def target_weights(self, signal, state):
        p = type(self).params
        vol = state["vol"].clip(lower=float(p["vol_floor"]))
        raw = np.sign(signal).fillna(0.0)
        scale = (float(p["vol_floor"]) / vol).clip(upper=1.0).fillna(0.0)
        return raw * scale / float(int(p["n_max"]))
'''


def _fit(selectable=True, reasons=None, **diag) -> FitnessResult:
    gates = GateResults(coverage_ok=selectable, degradation_ok=True,
                        deflation_ok=None, cost_ok=selectable,
                        reasons=reasons or {})
    base_diag = {"n_signals": 2, "mean_is_net_sharpe": 0.05,
                 "val_sharpe_dispersion": 0.01, "mean_turnover": 0.4,
                 "mean_activity": 0.8,
                 "per_signal": [{"signal": 0, "val_net_sharpe": 0.04,
                                 "capture": 0.7, "cost_x0.5": 0.05,
                                 "cost_x1.5": 0.03}],
                 "jitter_val_sharpes": [0.035, 0.045]}
    base_diag.update(diag)
    return FitnessResult(
        candidate_id="x", gates=gates,
        objective=ObjectiveVector(marginal_value=0.04, independence=0.035,
                                  robustness=0.7, parsimony=-20),
        diagnostics=base_diag)


# ── the deterministic brief ────────────────────────────────────────────────────

def test_brief_is_deterministic_and_carries_numbers():
    a, b = exec_mutation_brief(_fit()), exec_mutation_brief(_fit())
    assert a == b
    assert "0.040" in a          # mean net VAL Sharpe
    assert "per-signal" in a
    assert "ADVICE" in a


@pytest.mark.parametrize("reason_key,expect", [
    ("causality", "CAUSALITY FAILED"),
    ("turnover", "TURNOVER OVER CEILING"),
    ("activity", "BOOK TOO FLAT"),
    ("degradation", "IS→VAL DEGRADATION"),
    ("validity", "OUTPUT CONTRACT VIOLATED"),
])
def test_brief_advice_matches_failure_mode(reason_key, expect):
    brief = exec_mutation_brief(_fit(selectable=False,
                                     reasons={reason_key: "details"}))
    assert expect in brief


def test_brief_flags_cost_drag_and_coadaptation():
    fit = _fit()
    fit.objective.robustness = 0.2
    assert "COST DRAG" in exec_mutation_brief(fit)
    fit2 = _fit(val_sharpe_dispersion=0.2)
    assert "SIGNAL CO-ADAPTATION" in exec_mutation_brief(fit2)


# ── prompts + parsing ─────────────────────────────────────────────────────────

def test_mutation_prompt_contains_parent_brief_and_contract():
    seed = seed_execution_programs()[0]
    p = build_exec_mutation_prompt(seed, "THE BRIEF", "ex_child_1",
                                   ["taken_id"])
    assert seed.code in p
    assert "THE BRIEF" in p
    assert "THE EXECUTOR CONTRACT" in p
    assert 'executor_id "ex_child_1"' in p
    assert "taken_id" in p
    assert p == build_exec_mutation_prompt(seed, "THE BRIEF", "ex_child_1",
                                           ["taken_id"])  # deterministic


def test_crossover_prompt_contains_both_parents():
    a, b = seed_execution_programs()
    p = build_exec_crossover_prompt(a, "BA", b, "BB", "ex_child_2")
    assert a.code in p and b.code in p
    assert "BA" in p and "BB" in p


def test_parse_child_response_handles_fences_and_garbage():
    good = '{"executor_id": "e1", "code": "x = 1", "regime": "per_underlying"}'
    assert parse_exec_child_response(good)["executor_id"] == "e1"
    fenced = f"Here you go:\n```json\n{good}\n```\nGood luck!"
    assert parse_exec_child_response(fenced)["code"] == "x = 1"
    bad_regime = '{"executor_id": "e1", "code": "x", "regime": "sideways"}'
    assert parse_exec_child_response(bad_regime)["regime"] == "per_underlying"
    with pytest.raises(ValueError):
        parse_exec_child_response("no json here")
    with pytest.raises(ValueError):
        parse_exec_child_response('{"executor_id": "e1"}')  # no code


# ── the wired loop with a fake LLM ────────────────────────────────────────────

N_BARS, TICKERS = 480, ["A", "B", "C", "D", "E", "F"]

FACTOR_CODE = '''\
"""Test factor exm_mom."""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor


@register_factor
class FExmMom(BaseFactor):
    factor_id = "exm_mom"
    name = "exm_mom"
    category = "momentum"
    inputs = ["close"]
    prediction_horizon = 1

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        return close.pct_change().fillna(0.0)
'''


class FakeExecLLM:
    """Emits a valid executor child, echoing back the id the prompt demands."""

    def __init__(self):
        self.calls = 0
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        import re

        self.calls += 1
        self.prompts.append(prompt)
        m = re.search(r'executor_id "([a-z0-9_]+)"', prompt)
        child_id = m.group(1) if m else "ex_fake"
        code = CHILD_CODE.replace("CHILD_ID", child_id)
        import json
        return json.dumps({"executor_id": child_id, "name": "vol-scaled sign",
                           "regime": "per_underlying",
                           "mechanism": "size down when vol is high",
                           "expected_effect": "higher capture, lower drawdown",
                           "code": code})


@pytest.fixture()
def wired(tmp_path, monkeypatch):
    monkeypatch.setenv("QF_USE_MCP", "0")
    from quant_fund_agent.mcp import research_service as svc

    rng = np.random.default_rng(3)
    rets = np.zeros((N_BARS, len(TICKERS)))
    eps = rng.standard_normal((N_BARS, len(TICKERS))) * 0.01
    for t in range(1, N_BARS):
        rets[t] = 0.6 * rets[t - 1] + eps[t]
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="D")
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx,
                         columns=TICKERS)
    panel = {"close": close}

    monkeypatch.setattr(svc, "_load_panel_cached",
                        lambda data_dir, fields, n_tickers: panel)
    monkeypatch.setattr(svc, "_panel_cache_key",
                        lambda data_dir, fields, n_tickers: ("exm-panel",))
    svc._SIGNAL_CACHE.clear()

    out = svc.freeze_signals([{"factor_id": "exm_mom", "code": FACTOR_CODE}],
                             out_dir=str(tmp_path), version=1, target_horizon=1,
                             specs=[{"model": "ridge", "subset": [0]}])
    assert out["ok"], out.get("error")

    fake = FakeExecLLM()
    import quant_fund_agent.agents.execution_research.evolution.loop as loop_mod
    monkeypatch.setattr(loop_mod, "_get_llm",
                        lambda temperature, role=None: fake)

    cfg = ExecEvolutionRunConfig(
        out_dir=str(tmp_path / "evolution_exec"),
        signals_manifest=out["manifest_path"],
        generations=2, population_size=6, children_per_generation=3,
        seed=7, n_tickers=None,
        p_llm_semantic=0.6, p_crossover=0.2, p_jitter=0.2)
    return ExecEvolutionLoop(cfg), fake


def test_llm_children_are_admitted_with_briefs_in_prompts(wired):
    loop, fake = wired
    summary = loop.run()
    assert fake.calls > 0, "the LLM operator never fired"
    ops = {row["operator"] for row in loop.controller.lineage}
    assert "llm_semantic" in ops or "crossover" in ops
    # the prompt the LLM saw carried a real deterministic brief
    assert any("EXECUTION FITNESS" in p for p in fake.prompts)
    assert summary["archive"]


def test_invalid_llm_children_are_skipped_not_fatal(tmp_path, wired, monkeypatch):
    loop, fake = wired

    class BadLLM:
        calls = 0

        def invoke(self, prompt):
            BadLLM.calls += 1
            return '{"executor_id": "bad", "code": "import os"}'  # forbidden

    import quant_fund_agent.agents.execution_research.evolution.loop as loop_mod
    monkeypatch.setattr(loop_mod, "_get_llm",
                        lambda temperature, role=None: BadLLM())
    loop._llms.clear()
    summary = loop.run()
    assert summary["generations"] == 2         # the run completed regardless
    ops = {row["operator"] for row in loop.controller.lineage}
    assert "llm_semantic" not in ops           # nothing invalid was admitted
