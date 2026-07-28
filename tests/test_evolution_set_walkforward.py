"""P5 tests: SET-mode evaluation + loop, the CSCV PBO metric, and the
walk-forward final-validation driver.  All offline / LLM-free (SET operators
are structural; folds are seeded with prebuilt programs)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.comparison.config import ComparisonConfig
from quant_fund_agent.research_eval.deflation import pbo_cscv
from quant_fund_agent.research_eval.harness import EvalParams, evaluate_set

from tests.test_evolution_loop import (
    CHILD_BODY_1,
    MOM_BODY,
    VOL_BODY,
    _factor_code,
    _seeds,
)

N_BARS, TICKERS = 480, ["A", "B", "C", "D", "E", "F"]


def _panel(seed=3, rho=0.6):
    rng = np.random.default_rng(seed)
    rets = np.zeros((N_BARS, len(TICKERS)))
    eps = rng.standard_normal((N_BARS, len(TICKERS))) * 0.01
    for t in range(1, N_BARS):
        rets[t] = rho * rets[t - 1] + eps[t]
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="D")
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx,
                         columns=TICKERS)
    return {"close": close}


# ── PBO (CSCV) ────────────────────────────────────────────────────────────────

def test_pbo_separates_noise_from_skill():
    """Averaged over seeds: pure-luck selection ≈ coin flip OOS, genuine skill
    stays the winner (single-seed PBO is noisy with few candidates, so the
    assertion is on the seed-mean and the paired ordering)."""
    noise_pbos, skill_pbos = [], []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        M = rng.standard_normal((240, 10)) * 0.01           # pure luck
        out = pbo_cscv(M, n_groups=8)
        assert out["n_splits"] == math.comb(8, 4)
        noise_pbos.append(out["pbo"])
        M2 = M.copy()
        M2[:, 3] += 0.02                                    # one real edge
        skill_pbos.append(pbo_cscv(M2, n_groups=8)["pbo"])

    assert 0.2 <= float(np.mean(noise_pbos)) <= 0.8
    assert float(np.mean(skill_pbos)) < 0.1
    assert float(np.mean(skill_pbos)) < float(np.mean(noise_pbos))


def test_pbo_degenerate_inputs():
    assert pbo_cscv(np.zeros((100, 1)))["pbo"] is None      # one candidate
    assert pbo_cscv(np.zeros((4, 5)), n_groups=8)["pbo"] is None  # too few rows
    assert pbo_cscv(np.zeros((100, 5)), n_groups=7)["pbo"] is None  # odd groups


# ── SET-mode harness ─────────────────────────────────────────────────────────

def test_evaluate_set_axes_and_gates():
    panel = _panel()
    close = panel["close"]
    rng = np.random.default_rng(9)
    cfg = ComparisonConfig(preruns=["z"], target_horizon=1, seed=0)

    mom = close.pct_change().fillna(0.0)                       # real edge (AR panel)
    noise = pd.DataFrame(rng.standard_normal(close.shape),
                         index=close.index, columns=close.columns)
    codes = {"mom": "def calc():\n    return a + b\n",
             "noise": "def calc():\n    return a * b * c + 1\n"}

    r = evaluate_set({"mom": mom, "noise": noise}, panel, cfg,
                     params=EvalParams(stability_blocks=4),
                     member_codes=codes, candidate_id="pair")
    # axis 1: the set's OWN combined OOS IC (momentum member carries it)
    assert r.objective.marginal_value is not None
    assert r.objective.marginal_value > 0.1
    # axis 2: two ~independent members → internal PR/size near 1
    assert 0.5 < r.objective.independence <= 1.0
    # axis 4: parsimony = −(total complexity / size)
    assert r.objective.parsimony < 0
    assert r.diagnostics["set_size"] == 2
    assert r.gates.passed

    # a singleton set is trivially non-redundant
    r1 = evaluate_set({"mom": mom}, panel, cfg,
                      params=EvalParams(stability_blocks=4))
    assert r1.objective.independence == 1.0

    # a redundant pair scores lower on independence than the diverse pair
    r_dup = evaluate_set({"mom": mom, "mom2": mom * 1.0001}, panel, cfg,
                         params=EvalParams(stability_blocks=4))
    assert r_dup.objective.independence < r.objective.independence


# ── SET-mode loop (LLM-free: structural ops only) ────────────────────────────

@pytest.fixture()
def set_loop(monkeypatch, tmp_path):
    panel = _panel()
    monkeypatch.setenv("QF_USE_MCP", "0")

    from quant_fund_agent.mcp import research_service as svc

    monkeypatch.setattr(svc, "_load_panel_cached",
                        lambda data_dir, fields, n_tickers: panel)
    monkeypatch.setattr(svc, "_panel_cache_key",
                        lambda data_dir, fields, n_tickers: ("set-panel",))
    svc._SIGNAL_CACHE.clear()

    from quant_fund_agent.agents.factor_research.evolution.loop import (
        EvolutionLoop,
        EvolutionRunConfig,
    )

    cfg = EvolutionRunConfig(
        generations=3, population_size=5, children_per_generation=4,
        seed=17, unit="set", set_size=2, target_horizon=1,
        stability_blocks=4, n_tickers=None,
        out_dir=str(tmp_path / "evo_set"))
    loop = EvolutionLoop(cfg, data_context="CTX",
                         fields=["open", "high", "low", "close", "volume"])
    loop._load_known_ids = lambda: None
    return loop, tmp_path


def _four_seeds():
    return _seeds() + [
        __import__("quant_fund_agent.agents.factor_research.evolution.genome",
                   fromlist=["FactorProgram"]).FactorProgram(
            factor_id=fid, code=_factor_code(fid, body), prediction_horizon=1)
        for fid, body in [("seed_smooth", CHILD_BODY_1),
                          ("seed_rank", '(close.pct_change(3)).fillna(0.0)')]
    ]


def test_set_mode_loop_end_to_end(set_loop):
    loop, tmp_path = set_loop
    summary = loop.run(initial_programs=_four_seeds())

    assert summary["n_trials"] >= 2               # the seed sets were scored
    assert summary["archive"], "set archive should not be empty"
    for entry in summary["archive"]:
        assert entry["unit"] == "set"
        assert len(entry["factor_ids"]) >= 1

    ops = {row["operator"] for row in loop.controller.lineage
           if "event" not in row and row["generation"] > 0}
    assert ops <= {"structural", "splice", "member_jitter"}
    assert ops, "no set children were admitted"

    # state roundtrip preserves set genomes
    from quant_fund_agent.agents.factor_research.evolution.controller import (
        EvolutionController,
    )

    reloaded = EvolutionController.load(tmp_path / "evo_set" / "state.json")
    assert any(len(eg.genome.programs) > 1 for eg in reloaded.population())


# ── walk-forward final validation ─────────────────────────────────────────────

def test_walk_forward_validation(monkeypatch, tmp_path):
    panel = _panel()
    monkeypatch.setenv("QF_USE_MCP", "0")

    from quant_fund_agent.mcp import research_service as svc

    monkeypatch.setattr(svc, "_load_panel_cached",
                        lambda data_dir, fields, n_tickers: panel)
    monkeypatch.setattr(svc, "_panel_cache_key",
                        lambda data_dir, fields, n_tickers: ("wf-panel",))
    svc._SIGNAL_CACHE.clear()

    import quant_fund_agent.agents.factor_research.evolution.loop as loop_mod
    from quant_fund_agent.agents.factor_research.evolution.loop import (
        EvolutionRunConfig,
    )
    from quant_fund_agent.agents.factor_research.evolution.walkforward import (
        run_walk_forward,
    )

    monkeypatch.setattr(loop_mod.EvolutionLoop, "_build_data_context",
                        lambda self: "CTX")
    monkeypatch.setattr(loop_mod.EvolutionLoop, "_load_known_ids",
                        lambda self: None)

    # jitter-only children → the whole pass is LLM-free
    cfg = EvolutionRunConfig(
        generations=1, population_size=4, children_per_generation=2,
        seed=2, target_horizon=1, stability_blocks=4, n_tickers=None,
        p_llm_semantic=0.0, p_crossover=0.0, p_jitter=1.0,
        out_dir=str(tmp_path / "evo"))

    report = run_walk_forward(
        cfg, ["2024-11-01", "2025-01-01", "2025-03-01"],
        out_dir=tmp_path / "wf", initial_programs=_seeds())

    assert report["n_folds"] == 2
    for fold in report["folds"]:
        assert fold["oos"]["ok"], fold["oos"].get("error")
        assert fold["oos"]["n_test_bars"] > 0
        # the archive was discovered strictly before the scoring window
        assert fold["cutoff"] < fold["oos_end"]
    # the momentum seed generalises on the AR(1) panel → positive mean OOS IC
    assert report["mean_combined_oos_ic"] is not None
    assert report["mean_combined_oos_ic"] > 0.05
    assert (tmp_path / "wf" / "walkforward.json").exists()

    payload = json.loads((tmp_path / "wf" / "walkforward.json").read_text())
    assert payload["n_folds"] == 2


def test_walk_forward_rejects_bad_boundaries(tmp_path):
    from quant_fund_agent.agents.factor_research.evolution.loop import (
        EvolutionRunConfig,
    )
    from quant_fund_agent.agents.factor_research.evolution.walkforward import (
        run_walk_forward,
    )

    cfg = EvolutionRunConfig(out_dir=str(tmp_path))
    with pytest.raises(ValueError):
        run_walk_forward(cfg, ["2025-01-01"])
    with pytest.raises(ValueError):
        run_walk_forward(cfg, ["2025-02-01", "2025-01-01"])
