"""Tests for the mutation operators (window-jitter AST transforms, child parsing),
the reflection brief renderer, the in-memory factor compiler, and the harness's
plateau penalty — the deterministic half of the P1 evolutionary loop."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.agents.factor_research.evolution.genome import FactorProgram
from quant_fund_agent.agents.factor_research.evolution.mutation import (
    build_crossover_prompt,
    build_mutation_prompt,
    jitter_variants,
    jitter_windows,
    parse_child_response,
    random_jitter_child,
    rewrite_factor_id,
    window_constants,
)
from quant_fund_agent.agents.factor_research.evolution.reflection import mutation_brief
from quant_fund_agent.research_eval.fitness import (
    FitnessResult,
    GateResults,
    ObjectiveVector,
)

FACTOR_CODE = '''\
"""Momentum-vs-volatility test factor."""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, stddev, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class TestMomVol(BaseFactor):
    factor_id = "test_mom_vol"
    name = "Test momentum/vol"
    category = "momentum"
    inputs = ["close"]
    prediction_horizon = 6

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        mom = delta(close, 20)
        vol = stddev(close.pct_change(), 10)
        return ts_mean(mom / (vol + 1e-9), 5)
'''


# ── window scanning + jitter ─────────────────────────────────────────────────

def test_window_constants_found():
    assert sorted(window_constants(FACTOR_CODE)) == [5, 10, 20]


def test_jitter_windows_scales_and_counts():
    up, n_up = jitter_windows(FACTOR_CODE, 1.10)
    assert n_up == 3
    assert sorted(window_constants(up)) == [6, 11, 22]

    down, n_down = jitter_windows(FACTOR_CODE, 0.90)
    assert n_down == 3
    assert sorted(window_constants(down)) == [4, 9, 18]  # round(5·0.9)=4 (banker's)


def test_jitter_windows_always_changes_small_windows():
    code = "from quant_fund_agent.factors.ops import ts_mean\nx = ts_mean(y, 2)\n"
    up, n = jitter_windows(code, 1.05)  # 2*1.05 → rounds back to 2 → nudged to 3
    assert n == 1
    assert window_constants(up) == [3]


def test_jitter_ignores_non_window_ints():
    code = ("from quant_fund_agent.factors.ops import ts_mean\n"
            "x = ts_mean(y, 20) * 3 + 100\n")
    up, n = jitter_windows(code, 1.1)
    assert n == 1
    assert "* 3" in up and "100" in up  # arithmetic constants untouched


def test_jitter_variants_rewrites_probe_ids():
    prog = FactorProgram(factor_id="test_mom_vol", code=FACTOR_CODE)
    variants = jitter_variants(prog, scales=(0.9, 1.1))
    assert len(variants) == 2
    for vid, vcode in variants:
        assert vid.startswith("test_mom_vol_jit")
        assert f'factor_id = "{vid}"' in vcode.replace("'", '"')
        assert "test_mom_vol\"" not in vcode.replace("'", '"')


def test_jitter_variants_empty_for_windowless_factor():
    code = ("from quant_fund_agent.factors.ops import rank\n"
            "x = rank(y)\n")
    prog = FactorProgram(factor_id="xs_only", code=code)
    assert jitter_variants(prog) == []


def test_random_jitter_child_deterministic_under_seed():
    prog = FactorProgram(factor_id="test_mom_vol", code=FACTOR_CODE,
                         name="Test", trading_idea="momentum per unit vol")
    c1 = random_jitter_child(prog, np.random.default_rng(7), "child_a")
    c2 = random_jitter_child(prog, np.random.default_rng(7), "child_a")
    assert c1 is not None and c1.code == c2.code
    assert c1.factor_id == "child_a"
    assert window_constants(c1.code) != []
    assert c1.trading_idea == prog.trading_idea  # hypothesis carried over


def test_rewrite_factor_id_touches_only_string_constants():
    out = rewrite_factor_id(FACTOR_CODE, "test_mom_vol", "new_id")
    assert 'factor_id = \'new_id\'' in out or 'factor_id = "new_id"' in out
    assert "TestMomVol" in out  # class name (an identifier) untouched


# ── LLM child parsing ─────────────────────────────────────────────────────────

def test_parse_child_response_happy_path():
    content = """
    {"factor_id": "Child_One", "name": "Child", "category": "volatility",
     "trading_idea": "vol conditions momentum", "description": "d",
     "prediction_horizon": 12, "suggested_horizons": [6, 12],
     "expected_sign": -1, "code": "class X: pass"}
    """
    prog = parse_child_response(content)
    assert prog.factor_id == "child_one"        # lower-cased
    assert prog.prediction_horizon == 12
    assert prog.expected_sign == -1


def test_parse_child_response_rejects_missing_code():
    with pytest.raises(ValueError):
        parse_child_response('{"factor_id": "x", "code": ""}')


def test_parse_child_response_coerces_bad_sign_and_horizon():
    prog = parse_child_response(
        '{"factor_id": "x", "code": "y=1", "expected_sign": "up",'
        ' "prediction_horizon": -3}')
    assert prog.expected_sign is None
    assert prog.prediction_horizon == 6  # fallback default


# ── prompts ───────────────────────────────────────────────────────────────────

def test_mutation_prompt_contains_parent_and_brief():
    prog = FactorProgram(factor_id="test_mom_vol", code=FACTOR_CODE,
                         trading_idea="momentum per unit vol", expected_sign=1)
    p = build_mutation_prompt(prog, "VERDICT: failed degradation.",
                              "DATA CONTEXT HERE", ["a", "b"])
    assert "test_mom_vol" in p and "VERDICT: failed degradation." in p
    assert "DATA CONTEXT HERE" in p and "a, b" in p
    assert '"expected_sign"' in p  # child schema present


def test_crossover_prompt_contains_both_parents():
    pa = FactorProgram(factor_id="par_a", code="# code a")
    pb = FactorProgram(factor_id="par_b", code="# code b")
    p = build_crossover_prompt(pa, "brief a", pb, "brief b", "ctx", [])
    assert "par_a" in p and "par_b" in p
    assert "# code a" in p and "# code b" in p


# ── reflection brief ──────────────────────────────────────────────────────────

def _fitness(**diag) -> FitnessResult:
    base = {
        "standalone_ic": 0.004, "is_ic": 0.05, "val_ic": 0.004,
        "residual_ic": 0.01, "with_ic": 0.03, "base_ic": 0.01,
        "max_abs_corr": 0.8, "delta_participation": 0.1,
        "pr_before": 3.0, "pr_after": 3.1,
        "stability_ic_mean": 0.01, "stability_ic_std": 0.02,
        "stability_n_blocks": 4, "stability_score_kind": "fixed_marginal_delta",
        "stability_model": "gradient_boosting",
        "standalone_block_ic_mean": 0.0, "standalone_block_ic_std": 0.01,
        "standalone_n_blocks": 4,
        "sign_consistency": False, "coverage": 0.5,
        "degradation_ratio": 0.08,
        "ic_decay": {"1": 0.001, "6": 0.004, "24": 0.02},
        "plateau_penalty": 0.05, "jitter_ics": [0.001, -0.002],
        "deflation": {"deflated_t": -0.5, "luck_ic": 0.01},
        "n_trials": 123, "complexity": 14,
    }
    base.update(diag)
    return FitnessResult(
        candidate_id="c",
        objective=ObjectiveVector(marginal_value=0.02, independence=-0.3,
                                  robustness=-0.01, parsimony=-14.0),
        gates=GateResults(coverage_ok=True, degradation_ok=False, deflation_ok=True,
                          reasons={"degradation": "OOS/IS=0.080 < τ=0.5"}),
        diagnostics=base,
    )


def test_mutation_brief_is_deterministic_and_specific():
    fit = _fitness()
    brief1 = mutation_brief(fit, book_size=4)
    brief2 = mutation_brief(fit, book_size=4)
    assert brief1 == brief2
    # verdict names the failed gate
    assert "FAILED" in brief1 and "degradation" in brief1
    # all rule-based advice triggers fire on this diagnostics profile
    assert "correlated" in brief1                  # redundancy advice
    assert "conditioning/state variable" in brief1  # low-standalone/high-marginal
    assert "collapsed out-of-sample" in brief1     # degradation advice
    assert "knife-edge" in brief1                  # plateau advice
    assert "CONTRADICTS" in brief1                 # sign advice
    assert "24-bar horizon" in brief1              # decay re-anchoring advice
    assert "coverage" in brief1.lower()


def test_mutation_brief_clean_candidate_has_no_advice_section():
    fit = _fitness(max_abs_corr=0.1, standalone_ic=0.05, val_ic=0.05,
                   degradation_ratio=0.9, plateau_penalty=0.0,
                   sign_consistency=True, coverage=0.95,
                   ic_decay={"6": 0.05})
    fit.gates.degradation_ok = True
    fit.gates.reasons = {}
    brief = mutation_brief(fit)
    assert "WHAT TO DO DIFFERENTLY" not in brief
    assert "passed every hard gate" in brief


# ── in-memory compile + plateau penalty through the harness ──────────────────

def test_inmem_compile_and_signal_no_registry_leak():
    from quant_fund_agent.factors.inmem import compile_factor, signal_from_code
    from quant_fund_agent.factors.registry import get_factor_class

    cls = compile_factor(FACTOR_CODE, "test_mom_vol")
    assert get_factor_class("test_mom_vol") is None  # nothing leaked

    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    rng = np.random.default_rng(0)
    close = pd.DataFrame(100 + rng.standard_normal((120, 3)).cumsum(axis=0),
                         index=idx, columns=["A", "B", "C"])
    sig = signal_from_code(FACTOR_CODE, "test_mom_vol", {"close": close})
    assert sig.shape == close.shape


def test_inmem_compile_rejects_invalid_code():
    from quant_fund_agent.agents.factor_research.codegen import CodeValidationError
    from quant_fund_agent.factors.inmem import compile_factor

    with pytest.raises(CodeValidationError):
        compile_factor("import os\n", "evil")


def test_plateau_penalty_docks_knife_edge_candidate():
    """A candidate whose jitter variants lose the edge gets a lower robustness."""
    from quant_fund_agent.comparison.config import ComparisonConfig
    from quant_fund_agent.research_eval.harness import EvalParams, evaluate_candidate

    n, tickers = 600, ["A", "B", "C", "D", "E"]
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="min")
    rets = rng.standard_normal((n, len(tickers))) * 0.01
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=tickers)
    panel = {"close": close}
    cfg = ComparisonConfig(preruns=["z"], target_horizon=1, seed=0)

    fwd = close.pct_change().shift(-1)
    good = fwd + 0.002 * rng.standard_normal((n, len(tickers)))
    noise_jitters = [pd.DataFrame(rng.standard_normal((n, len(tickers))),
                                  index=idx, columns=tickers) for _ in range(2)]

    with_pen = evaluate_candidate(good, [], panel, cfg, params=EvalParams(),
                                  jitter_signals=noise_jitters)
    without = evaluate_candidate(good, [], panel, cfg, params=EvalParams())

    assert with_pen.diagnostics["plateau_penalty"] > 0.1
    assert with_pen.objective.robustness < without.objective.robustness
    assert without.diagnostics["plateau_penalty"] is None

    # plateau-stable variants (≈ the candidate itself) → ~zero penalty
    stable = evaluate_candidate(good, [], panel, cfg, params=EvalParams(),
                                jitter_signals=[good, good])
    assert stable.diagnostics["plateau_penalty"] == pytest.approx(0.0, abs=1e-9)
