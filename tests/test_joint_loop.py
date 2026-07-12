"""Tests for the joint outer loop (J0/J1): sequential-equals-standalone,
round-robin with re-freeze + rescore, ledger invariants, resume.

Both arms run jitter-only (no LLM anywhere), in-process (QF_USE_MCP=0), on a
synthetic AR(1) panel.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.agents.execution_research.evolution.loop import (
    ExecEvolutionRunConfig,
)
from quant_fund_agent.agents.factor_research.evolution.genome import FactorProgram
from quant_fund_agent.agents.factor_research.evolution.loop import (
    EvolutionLoop,
    EvolutionRunConfig,
)
from quant_fund_agent.joint_evolution.loop import JointEvolutionLoop, JointRunConfig

N_BARS, TICKERS = 480, ["A", "B", "C", "D", "E", "F"]


def _factor_code(fid: str, body: str) -> str:
    return f'''\
"""Test factor {fid}."""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import stddev, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class F_{fid}(BaseFactor):
    factor_id = "{fid}"
    name = "{fid}"
    category = "momentum"
    inputs = ["close"]
    prediction_horizon = 1

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        return {body}
'''


def _seeds() -> list[FactorProgram]:
    # both carry WINDOW constants so the jitter-only factor arm can mutate them
    return [
        FactorProgram(factor_id="jt_mom",
                      code=_factor_code("jt_mom", "ts_mean(close.pct_change(), 3).fillna(0.0)"),
                      name="mom", category="momentum", expected_sign=1,
                      prediction_horizon=1),
        FactorProgram(factor_id="jt_vol",
                      code=_factor_code("jt_vol", "stddev(close.pct_change(), 12).fillna(0.0)"),
                      name="vol", category="volatility", expected_sign=1,
                      prediction_horizon=1),
    ]


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
                        lambda data_dir, fields, n_tickers: ("joint-panel",))
    monkeypatch.setattr(EvolutionLoop, "_load_known_ids", lambda self: None)
    svc._SIGNAL_CACHE.clear()
    return tmp_path


def _factor_cfg(out_dir, generations=2) -> EvolutionRunConfig:
    return EvolutionRunConfig(
        generations=generations, population_size=6, children_per_generation=3,
        seed=11, target_horizon=1, cpcv_groups=4, cpcv_k=1, n_tickers=None,
        p_llm_semantic=0.0, p_crossover=0.0, p_jitter=1.0,
        out_dir=str(out_dir))


def _exec_cfg(out_dir) -> ExecEvolutionRunConfig:
    return ExecEvolutionRunConfig(
        out_dir=str(out_dir), signals_manifest="(joint)",
        generations=2, population_size=6, children_per_generation=3,
        seed=7, n_tickers=None)


def _archive_invariants(state_path) -> dict:
    state = json.loads(state_path.read_text())
    return {
        "n_trials": state["n_trials"],
        "generation": state["generation"],
        "archive": sorted(
            (tuple(eg["genome"]["programs"][0]["factor_id"].split()),
             json.dumps(eg["fitness"]["objective"], sort_keys=True))
            for eg in state["archive"]),
        "fingerprints": sorted(state["fingerprints"]),
    }


def test_sequential_factor_block_equals_standalone_run(wired):
    """THE key J0 regression: the joint layer's factor block is byte-equivalent
    to a standalone run (same seeds, same rng) up to the uuid in genome_id —
    compared on n_trials / archive objectives / dedup fingerprints."""
    tmp_path = wired

    standalone = EvolutionLoop(_factor_cfg(tmp_path / "standalone"),
                               data_context="CTX",
                               fields=["open", "high", "low", "close", "volume"])
    standalone.run(initial_programs=_seeds())

    joint = JointEvolutionLoop(
        JointRunConfig(out_dir=str(tmp_path / "joint"), total_blocks=1,
                       gens_per_block=2, scheduler="sequential",
                       target_horizon=1, n_tickers=None),
        _factor_cfg(tmp_path / "ignored"), _exec_cfg(tmp_path / "ignored2"),
        data_context="CTX")
    joint.run(initial_factor_programs=_seeds())

    a = _archive_invariants(tmp_path / "standalone" / "state.json")
    b = _archive_invariants(tmp_path / "joint" / "factor" / "state.json")
    assert a == b


def test_round_robin_freezes_rescore_and_ledger_invariants(wired):
    tmp_path = wired
    joint = JointEvolutionLoop(
        JointRunConfig(out_dir=str(tmp_path / "jr"), total_blocks=3,
                       gens_per_block=1, scheduler="round_robin",
                       target_horizon=1, n_tickers=None),
        _factor_cfg(tmp_path / "x"), _exec_cfg(tmp_path / "y"),
        data_context="CTX")
    summary = joint.run(initial_factor_programs=_seeds())

    assert summary["arm_choices"] == ["factor", "exec", "factor"]
    assert summary["blocks"] == 3
    # two factor blocks → two freezes; the second re-scored the exec archive
    assert summary["frozen_signals_version"] == 2
    assert summary["sota_executor"] is not None
    assert summary["J"] is not None

    rows = [json.loads(l) for l in
            (tmp_path / "jr" / "blocks.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert rows[0]["boundary"]["refrozen"] is True
    assert rows[2]["boundary"]["rescore"]["rescored"] > 0

    led = summary["ledger"]
    assert led["n_factor"] > 0 and led["n_exec"] > 0
    # looks ≥ families + rescores + one objective score per block
    assert led["n_joint_looks"] >= led["n_factor"] + led["n_exec"] + 3

    # exec family count must NOT include the re-scores
    n_rescored = rows[2]["boundary"]["rescore"]["rescored"]
    assert led["n_exec"] == rows[1]["candidates_scored"]
    assert n_rescored > 0


def test_joint_resume_runs_only_remaining_blocks(wired):
    tmp_path = wired
    cfg2 = JointRunConfig(out_dir=str(tmp_path / "jr2"), total_blocks=2,
                          gens_per_block=1, scheduler="round_robin",
                          target_horizon=1, n_tickers=None)
    j1 = JointEvolutionLoop(cfg2, _factor_cfg(tmp_path / "x"),
                            _exec_cfg(tmp_path / "y"), data_context="CTX")
    s1 = j1.run(initial_factor_programs=_seeds())
    assert s1["blocks"] == 2
    looks_after_2 = s1["ledger"]["n_joint_looks"]

    # extend the schedule by one block; a fresh loop resumes at block 2
    cfg3 = JointRunConfig(**{**cfg2.to_dict(), "total_blocks": 3})
    j2 = JointEvolutionLoop(cfg3, _factor_cfg(tmp_path / "x"),
                            _exec_cfg(tmp_path / "y"), data_context="CTX")
    s2 = j2.run()
    assert s2["blocks"] == 3
    assert s2["arm_choices"] == ["factor", "exec", "factor"]  # history preserved
    assert s2["ledger"]["n_joint_looks"] > looks_after_2

    # exec block scheduled first without a freeze must be impossible
    with pytest.raises(RuntimeError, match="factor block"):
        bad = JointEvolutionLoop(
            JointRunConfig(out_dir=str(tmp_path / "bad"), total_blocks=1,
                           gens_per_block=1, scheduler="round_robin",
                           target_horizon=1, n_tickers=None),
            _factor_cfg(tmp_path / "x"), _exec_cfg(tmp_path / "y"))
        bad.sota.block_index = 1          # force a non-zero start
        bad.cfg.total_blocks = 2
        bad.run()
