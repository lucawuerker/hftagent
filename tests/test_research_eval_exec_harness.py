"""Tests for the execution fitness harness (E0) — the exec reward channel.

Mirrors the factor harness leak tests: dev-slice poison invariance, the
last-dev-bar drop, and the truncation-replay causality probe catching a
planted look-ahead executor.  Plus gates, axes and the MCP service seam.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.execution.base import BaseExecutor, get_executor
from quant_fund_agent.research_eval.exec_harness import (
    ExecEvalParams,
    causality_probe,
    evaluate_executor,
)
from quant_fund_agent.research_eval.splits import ThreeWaySplit, three_way_split

N_BARS, TICKERS = 480, ["A", "B", "C", "D", "E", "F"]


def _close_full(seed=3):
    """AR(1) returns → a momentum signal has genuine edge."""
    rng = np.random.default_rng(seed)
    rets = np.zeros((N_BARS, len(TICKERS)))
    eps = rng.standard_normal((N_BARS, len(TICKERS))) * 0.01
    for t in range(1, N_BARS):
        rets[t] = 0.6 * rets[t - 1] + eps[t]
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="D")
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=TICKERS)


def _dev_slice(close_full: pd.DataFrame, signal_full: pd.DataFrame):
    """Replicate the service prologue: dev-sliced panel/signal + relative split."""
    full = three_way_split(close_full.index, is_frac=0.6, val_frac=0.2)
    dev = full.is_val_mask
    panel = {"close": close_full.loc[dev]}
    split = ThreeWaySplit(is_mask=full.is_mask[dev], val_mask=full.val_mask[dev],
                          test_mask=full.test_mask[dev])
    return panel, [signal_full.loc[dev]], split


class _SignExec(BaseExecutor):
    executor_id = "sign_test_exec"
    regime = "per_underlying"

    def target_weights(self, signal, state):
        return np.sign(signal).fillna(0.0) / signal.shape[1]


class _FullSampleZExec(BaseExecutor):
    """Planted LEAK: full-sample standardisation reads the entire window."""

    executor_id = "leaky_test_exec"
    regime = "per_underlying"

    def target_weights(self, signal, state):
        z = (signal - signal.mean()) / signal.std()
        return z.clip(-1.0, 1.0).fillna(0.0) / signal.shape[1]


class _FlatExec(BaseExecutor):
    executor_id = "flat_test_exec"
    regime = "per_underlying"

    def target_weights(self, signal, state):
        return signal * 0.0


@pytest.fixture()
def fixtures():
    close = _close_full()
    signal = close.pct_change().fillna(0.0)      # momentum → real edge under AR(1)
    return close, signal


# ── axes + gates on a genuinely predictive setup ───────────────────────────────

def test_predictive_executor_scores_positive_and_passes_gates(fixtures):
    close, signal = fixtures
    panel, frozen, split = _dev_slice(close, signal)
    res = evaluate_executor(_SignExec(), frozen, panel, split,
                            params=ExecEvalParams(),
                            candidate_code="w = sign(s) / n", candidate_id="sign")
    assert res.candidate_id == "sign"
    assert res.objective.marginal_value is not None
    assert res.objective.marginal_value > 0            # net VAL Sharpe positive
    assert res.objective.independence is not None      # cross-signal generalisation
    assert res.objective.robustness is not None        # net÷gross capture
    assert res.objective.robustness <= 1.0
    assert res.objective.parsimony is not None
    assert res.objective.structural_novelty is None    # no archive passed
    assert res.gates.coverage_ok is True
    assert res.gates.cost_ok is True                   # causality probe passed
    assert res.gates.deflation_ok is None              # WS1: publish-time control
    assert res.selectable
    assert res.diagnostics["causality_probe"]["passed"] is True
    assert res.diagnostics["mean_turnover"] > 0
    assert 0 < res.diagnostics["mean_activity"] <= 1
    # ±50% cost sensitivity diagnostic present per signal
    assert "cost_x0.5" in res.diagnostics["per_signal"][0]


def test_last_dev_bar_dropped_from_scoring(fixtures):
    close, signal = fixtures
    panel, frozen, split = _dev_slice(close, signal)
    res = evaluate_executor(_SignExec(), frozen, panel, split)
    # the last dev bar's forward return would need a TEST price → NaN → dropped
    assert res.diagnostics["per_signal"][0]["val_n_obs"] == int(split.val_mask.sum()) - 1


def test_poison_invariance_of_the_dev_slice(fixtures):
    """Corrupt every TEST row of panel+signal → the fitness dict must be identical."""
    close, signal = fixtures
    full = three_way_split(close.index, is_frac=0.6, val_frac=0.2)

    poisoned_close, poisoned_signal = close.copy(), signal.copy()
    rows = np.flatnonzero(full.test_mask)
    poisoned_close.iloc[rows] = 1e9
    poisoned_signal.iloc[rows] = -1e9

    panel_a, frozen_a, split_a = _dev_slice(close, signal)
    res_a = evaluate_executor(_SignExec(), frozen_a, panel_a, split_a)
    panel_b, frozen_b, split_b = _dev_slice(poisoned_close, poisoned_signal)
    res_b = evaluate_executor(_SignExec(), frozen_b, panel_b, split_b)
    assert res_a.to_dict() == res_b.to_dict()


# ── the causality probe ────────────────────────────────────────────────────────

def test_causality_probe_passes_honest_executor(fixtures):
    close, signal = fixtures
    panel, frozen, _ = _dev_slice(close, signal)
    probe = causality_probe(_SignExec(), frozen[0], panel)
    assert probe["passed"] is True


def test_causality_probe_catches_full_sample_standardisation(fixtures):
    close, signal = fixtures
    panel, frozen, split = _dev_slice(close, signal)
    probe = causality_probe(_FullSampleZExec(), frozen[0], panel)
    assert probe["passed"] is False

    res = evaluate_executor(_FullSampleZExec(), frozen, panel, split)
    assert res.gates.cost_ok is False
    assert "causality" in res.gates.reasons
    assert not res.selectable


def test_causality_probe_catches_leaky_seed_variant(fixtures):
    """The per-underlying seed with zscore_basis='full' IS a leak — prove it."""
    close, signal = fixtures
    panel, frozen, _ = _dev_slice(close, signal)
    cls = get_executor("zscore_threshold_equal_weight")
    leaky = cls()
    leaky.zscore_basis = "full"
    probe = causality_probe(leaky, frozen[0], panel)
    assert probe["passed"] is False


# ── remaining gates ────────────────────────────────────────────────────────────

def test_flat_executor_fails_min_activity(fixtures):
    close, signal = fixtures
    panel, frozen, split = _dev_slice(close, signal)
    res = evaluate_executor(_FlatExec(), frozen, panel, split)
    assert res.gates.coverage_ok is False
    assert "activity" in res.gates.reasons
    assert not res.selectable


def test_turnover_ceiling_gate(fixtures):
    close, signal = fixtures
    panel, frozen, split = _dev_slice(close, signal)
    res = evaluate_executor(_SignExec(), frozen, panel, split,
                            params=ExecEvalParams(gate_turnover=1e-6))
    assert res.gates.cost_ok is False
    assert "turnover" in res.gates.reasons


def test_selection_deflation_on_evaluates_the_gate(fixtures):
    close, signal = fixtures
    panel, frozen, split = _dev_slice(close, signal)
    res = evaluate_executor(_SignExec(), frozen, panel, split,
                            params=ExecEvalParams(selection_deflation="on",
                                                  n_trials=50))
    assert res.gates.deflation_ok is not None
    assert res.diagnostics["deflated_sharpe_prob"] is not None


def test_crashing_executor_is_a_scored_failure(fixtures):
    close, signal = fixtures
    panel, frozen, split = _dev_slice(close, signal)

    class Boom(BaseExecutor):
        executor_id = "boom_test_exec"
        regime = "per_underlying"

        def target_weights(self, signal, state):
            raise RuntimeError("kaboom")

    res = evaluate_executor(Boom(), frozen, panel, split)
    assert res.gates.coverage_ok is False
    assert not res.selectable
    assert "kaboom" in res.diagnostics["error"]


def test_structural_novelty_vs_archived_executors(fixtures):
    """A clone of an archived executor scores 0 novelty; fresh code scores high."""
    close, signal = fixtures
    panel, frozen, split = _dev_slice(close, signal)
    code = "def target_weights(s): return s.rolling(5).mean().pipe(lambda z: z/9)"
    clone = evaluate_executor(_SignExec(), frozen, panel, split,
                              candidate_code=code, archive_codes=[code])
    assert clone.objective.structural_novelty == pytest.approx(0.0)
    fresh = evaluate_executor(_SignExec(), frozen, panel, split,
                              candidate_code="w = -vol * drawdown_scaled_book(x)",
                              archive_codes=[code])
    assert fresh.objective.structural_novelty > 0.3


def test_jitter_variants_reported_as_diagnostic(fixtures):
    close, signal = fixtures
    panel, frozen, split = _dev_slice(close, signal)
    res = evaluate_executor(_SignExec(), frozen, panel, split,
                            jitter_executors=[_SignExec(), _FlatExec()])
    js = res.diagnostics["jitter_val_sharpes"]
    assert len(js) == 2
    assert js[0] == pytest.approx(res.diagnostics["per_signal"][0]["val_net_sharpe"])


# ── cross-signal aggregation ───────────────────────────────────────────────────

def test_cross_signal_dispersion_penalises_inconsistency(fixtures):
    close, signal = fixtures
    panel, frozen, split = _dev_slice(close, signal)
    noise = pd.DataFrame(
        np.random.default_rng(9).normal(0, 1, frozen[0].shape),
        index=frozen[0].index, columns=frozen[0].columns)
    res2 = evaluate_executor(_SignExec(), [frozen[0], noise], panel, split)
    assert res2.diagnostics["n_signals"] == 2
    assert res2.diagnostics["val_sharpe_dispersion"] > 0
    assert res2.objective.independence < res2.objective.marginal_value


# ── the MCP service seam (in-process) ──────────────────────────────────────────

FACTOR_CODE = '''\
"""Test factor svc_mom."""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor


@register_factor
class FSvcMom(BaseFactor):
    factor_id = "svc_mom"
    name = "svc_mom"
    category = "momentum"
    inputs = ["close"]
    prediction_horizon = 1

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        return close.pct_change().fillna(0.0)
'''

EXEC_CODE = '''\
"""Test executor sign_exec_svc."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.execution.base import BaseExecutor, register_executor


@register_executor
class SignExecSvc(BaseExecutor):
    executor_id = "sign_exec_svc"
    name = "sign"
    regime = "per_underlying"
    inputs = ["signal"]
    params = {"scale": 6.0}

    def target_weights(self, signal, state):
        return np.sign(signal).fillna(0.0) / 6.0
'''


def test_service_freeze_then_evaluate_in_process(tmp_path, monkeypatch, fixtures):
    """The full E0 seam: freeze_signals → evaluate_executor_fitness, no MCP."""
    from quant_fund_agent.mcp import research_service as svc

    close, _ = fixtures
    monkeypatch.setattr(svc, "_load_panel_cached",
                        lambda data_dir, fields, n_tickers: {"close": close})
    monkeypatch.setenv("QF_USE_MCP", "0")

    frozen = svc.freeze_signals(
        [{"factor_id": "svc_mom", "code": FACTOR_CODE}],
        out_dir=str(tmp_path), version=1, target_horizon=1,
        specs=[{"model": "ridge", "subset": [0]}])
    assert frozen["ok"], frozen.get("error")
    assert frozen["manifest"]["poison_audit"]["passed"] is True

    out = svc.evaluate_executor_fitness(
        {"executor_id": "sign_exec_svc", "code": EXEC_CODE},
        frozen["manifest_path"])
    assert out["ok"], out.get("error")
    fit = out["fitness"]
    assert fit["candidate_id"] == "sign_exec_svc"
    assert fit["objective"]["marginal_value"] is not None
    assert fit["gates"]["passed"] is True
    assert fit["raw"]["frozen_signals_version"] == 1

    bad = svc.evaluate_executor_fitness(
        {"executor_id": "sign_exec_svc", "code": "import os"},
        frozen["manifest_path"])
    assert bad["ok"] is False


def test_client_routes_in_process(tmp_path, monkeypatch, fixtures):
    from quant_fund_agent.mcp import research_client, research_service as svc

    close, _ = fixtures
    monkeypatch.setattr(svc, "_load_panel_cached",
                        lambda data_dir, fields, n_tickers: {"close": close})
    monkeypatch.setenv("QF_USE_MCP", "0")

    frozen = research_client.freeze_signals(
        [{"factor_id": "svc_mom", "code": FACTOR_CODE}],
        out_dir=str(tmp_path), version=1, target_horizon=1,
        specs=[{"model": "ridge", "subset": [0]}])
    assert frozen["ok"]
    out = research_client.evaluate_executor_fitness(
        {"executor_id": "sign_exec_svc", "code": EXEC_CODE},
        frozen["manifest_path"])
    assert out["ok"]
