"""Tests for the BaseExecutor contract, registry, weight validation and BookState (E0)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.execution.base import (
    EXECUTOR_REGISTRY,
    BaseExecutor,
    BookState,
    get_executor,
    list_executors,
    register_executor,
    run_executor,
    validate_weights,
)
from quant_fund_agent.execution.state import build_state_frames

N, T = 4, 60


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=T, freq="D")
    cols = [f"S{i}" for i in range(N)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (T, N)), axis=0),
                         index=idx, columns=cols)
    signal = pd.DataFrame(rng.normal(0, 1, (T, N)), index=idx, columns=cols)
    panel = {"close": close}
    return signal, build_state_frames(panel, signal), close


# ── registry ───────────────────────────────────────────────────────────────────

def test_register_and_get():
    class Tmp(BaseExecutor):
        executor_id = "tmp_reg_test"

        def target_weights(self, signal, state):
            return signal * 0.0

    try:
        register_executor(Tmp)
        assert get_executor("tmp_reg_test") is Tmp
        assert "tmp_reg_test" in list_executors()
        register_executor(Tmp)  # idempotent for the same class

        class Clash(BaseExecutor):
            executor_id = "tmp_reg_test"

        with pytest.raises(ValueError, match="already registered"):
            register_executor(Clash)
    finally:
        EXECUTOR_REGISTRY.pop("tmp_reg_test", None)


def test_get_unknown_raises():
    with pytest.raises(KeyError, match="unknown executor_id"):
        get_executor("does_not_exist_xyz")


def test_seeds_are_registered():
    assert {"topk_dollar_neutral", "zscore_threshold_equal_weight"} <= set(list_executors())


# ── weight contract ────────────────────────────────────────────────────────────

def _wframe(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.DataFrame(values, index=idx, columns=[f"S{i}" for i in range(len(values[0]))])


def test_validate_weights_ok_and_nan_allowed():
    w = _wframe([[0.5, -0.5, np.nan, 0.0], [0.25, -0.25, 0.25, -0.25]])
    assert validate_weights(w, "cross_sectional") == []


def test_validate_weights_inf_fails():
    w = _wframe([[np.inf, 0.0, 0.0, 0.0]])
    assert any("inf" in p for p in validate_weights(w, "per_underlying"))


def test_validate_weights_per_name_bound():
    w = _wframe([[1.5, 0.0, 0.0, 0.0]])
    assert any("per-name" in p for p in validate_weights(w, "per_underlying"))


def test_validate_weights_gross_bound():
    w = _wframe([[0.9, -0.9, 0.9, -0.9]])  # gross 3.6 > 2.0
    assert any("gross-leverage" in p for p in validate_weights(w, "per_underlying"))


def test_validate_weights_neutrality_only_cross_sectional():
    w = _wframe([[0.9, 0.9, 0.0, 0.0]])  # net 1.8
    assert any("neutrality" in p for p in validate_weights(w, "cross_sectional"))
    assert validate_weights(w, "per_underlying") == []


# ── BookState ──────────────────────────────────────────────────────────────────

def test_book_state_tracks_pnl_and_drawdown():
    cols = pd.Index(["A", "B"])
    book = BookState(cols)
    p0 = pd.Series([100.0, 100.0], index=cols)
    p1 = pd.Series([110.0, 100.0], index=cols)   # A +10%
    p2 = pd.Series([99.0, 100.0], index=cols)    # A −10%

    book.mark(p0)
    book.rebalance(pd.Series([1.0, 0.0], index=cols))   # long A at 100
    book.mark(p1)
    assert book.equity == pytest.approx(0.10)
    assert book.unrealised_pnl["A"] == pytest.approx(0.10)
    assert book.drawdown == pytest.approx(0.0)
    book.rebalance(pd.Series([1.0, 0.0], index=cols))   # hold
    book.mark(p2)
    assert book.equity == pytest.approx(0.10 - 0.10)
    assert book.drawdown == pytest.approx(-0.10)
    assert book.unrealised_pnl["A"] == pytest.approx(-0.01)  # entry stays 100


# ── stepwise vs vectorised driver ──────────────────────────────────────────────

class _SignVectorised(BaseExecutor):
    executor_id = "sign_vec_test"
    regime = "per_underlying"

    def target_weights(self, signal, state):
        return np.sign(signal).fillna(0.0) / signal.shape[1]


class _SignStepwise(BaseExecutor):
    executor_id = "sign_step_test"
    regime = "per_underlying"

    def step(self, t, signal_row, state_row, book):
        return np.sign(signal_row).fillna(0.0) / len(signal_row)


def test_stepwise_matches_vectorised_for_path_independent_program():
    signal, state, close = _inputs()
    wv = run_executor(_SignVectorised(), signal, state, close)
    ws = run_executor(_SignStepwise(), signal, state, close)
    pd.testing.assert_frame_equal(wv, ws)


def test_stepwise_needs_close():
    signal, state, _ = _inputs()
    with pytest.raises(ValueError, match="needs `close`"):
        run_executor(_SignStepwise(), signal, state, None)


def test_stepwise_book_is_visible_to_step():
    """A path-dependent program (trade once, then hold) needs the book state."""

    class HoldFirst(BaseExecutor):
        executor_id = "hold_first_test"
        regime = "per_underlying"

        def step(self, t, signal_row, state_row, book):
            if (book.positions.abs() > 0).any():
                return book.positions          # hold whatever we own
            return pd.Series(1.0 / len(signal_row), index=signal_row.index)

    signal, state, close = _inputs()
    w = run_executor(HoldFirst(), signal, state, close)
    # first bar enters, every later bar holds the identical book
    assert (w.iloc[0] > 0).all()
    assert (w.diff().abs().iloc[1:].to_numpy() == 0).all()
