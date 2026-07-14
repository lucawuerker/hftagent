"""End-to-end test of the GP factor-mining loop on a synthetic panel.

Runs fully in-process (no MCP subprocess, no LLM): the panel loader is
monkeypatched to an AR(1)-momentum panel so a momentum tree has real edge, and
the whole path is exercised — seed → evaluate (research_eval harness) → NSGA-II
insert → GP mutate/crossover → dedup → archive → checkpoint.  Also checks that
archived GP factors compile and yield non-degenerate signals (the property the
comparison harness's ``usable_factor_ids`` gate requires).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.agents.factor_research.gp.grammar import (
    Node,
    build_grammar,
    random_tree,
)
from quant_fund_agent.agents.factor_research.gp.loop import GPLoop, GPRunConfig
from quant_fund_agent.factors import inmem

N_BARS, TICKERS = 480, ["A", "B", "C", "D", "E", "F"]


@pytest.fixture()
def ar1_panel():
    """AR(1)-momentum returns → a 1-bar momentum factor has genuine edge."""
    rng = np.random.default_rng(3)
    rets = np.zeros((N_BARS, len(TICKERS)))
    eps = rng.standard_normal((N_BARS, len(TICKERS))) * 0.01
    for t in range(1, N_BARS):
        rets[t] = 0.6 * rets[t - 1] + eps[t]
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="D")
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=TICKERS)
    return {"close": close}


@pytest.fixture()
def wired(ar1_panel, tmp_path, monkeypatch):
    monkeypatch.setenv("QF_USE_MCP", "0")
    from quant_fund_agent.mcp import research_service as svc

    monkeypatch.setattr(svc, "_load_panel_cached",
                        lambda data_dir, fields, n_tickers: ar1_panel)
    monkeypatch.setattr(svc, "_panel_cache_key",
                        lambda data_dir, fields, n_tickers: ("gp-test-panel",))
    svc._SIGNAL_CACHE.clear()

    cfg = GPRunConfig(
        generations=2, population_size=6, children_per_generation=4,
        seed=11, depth_schedule=(3,), target_horizon=1,
        is_frac=0.5, val_frac=0.25, stability_blocks=4,
        marginal_model="ridge", n_tickers=None,
        out_dir=str(tmp_path / "gp"),
    )
    loop = GPLoop(cfg, allowed_fields=["close"])
    loop._load_known_ids = lambda: None  # hermetic
    return loop, tmp_path


def _seed_trees(grammar):
    """A guaranteed-good momentum seed + returns + a couple of random trees."""
    rng = np.random.default_rng(0)
    momentum = Node("op", "delta", (Node("field", "close"), Node("window", 1)))
    return [
        momentum,
        Node("returns"),
        random_tree(grammar, rng, 3, method="grow"),
        random_tree(grammar, rng, 3, method="full"),
    ]


def test_full_gp_loop_runs_and_checkpoints(wired):
    loop, tmp_path = wired
    summary = loop.run(initial_trees=_seed_trees(loop.grammar))

    assert summary["engine"] == "gp"
    assert summary["generations"] == 2
    assert summary["n_trials"] >= 2                 # seeds were scored + billed
    assert summary["archive"], "a momentum seed should reach the archive"

    # checkpoints exist and reload cleanly
    state_path = tmp_path / "gp" / "state.json"
    assert state_path.exists()
    from quant_fund_agent.agents.factor_research.evolution.controller import (
        EvolutionController,
    )

    reloaded = EvolutionController.load(state_path)
    assert reloaded.n_trials == summary["n_trials"]

    # lineage rows are all tagged with GP operators
    lineage = [json.loads(l) for l in
               (tmp_path / "gp" / "lineage.jsonl").read_text().splitlines()]
    assert lineage
    assert all(row["operator"].startswith("gp_") for row in lineage)
    # at least one non-seed GP operator actually fired
    assert any(row["operator"] != "gp_seed" for row in lineage)


def test_archived_factors_compile_and_are_nondegenerate(wired, ar1_panel):
    """Every archived GP factor is a real, computable, non-degenerate factor."""
    loop, _ = wired
    summary = loop.run(initial_trees=_seed_trees(loop.grammar))
    assert summary["archive"]

    for eg in loop.controller.archive:
        prog = eg.genome.program
        sig = inmem.signal_from_code(prog.code, prog.factor_id, ar1_panel)
        arr = np.asarray(sig.to_numpy(), dtype="float64")
        finite = np.isfinite(arr)
        assert finite.sum() >= 5
        assert float(np.std(arr[finite])) > 1e-12
        # provenance: horizon carried onto the program
        assert prog.prediction_horizon == 1


def test_random_seeding_path_runs(wired, tmp_path):
    """The random-seed path (no initial_trees) runs end-to-end, no LLM involved."""
    loop, _ = wired
    cfg = GPRunConfig(
        generations=1, population_size=6, children_per_generation=3,
        seed=1, seed_pop=10, depth_schedule=(3,), target_horizon=1,
        is_frac=0.5, val_frac=0.25, stability_blocks=4,
        marginal_model="ridge", n_tickers=None, out_dir=str(tmp_path / "gp2"),
    )
    fresh = GPLoop(cfg, allowed_fields=["close"])
    fresh._load_known_ids = lambda: None
    summary = fresh.run()
    assert summary["engine"] == "gp"
    assert summary["n_trials"] >= 1
    assert summary["population"] >= 1
