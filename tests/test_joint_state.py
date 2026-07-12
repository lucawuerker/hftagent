"""Tests for SOTAState + TrialsLedger (J0)."""

from __future__ import annotations

import pytest

from quant_fund_agent.joint_evolution.ledger import TrialsLedger
from quant_fund_agent.joint_evolution.state import SOTAState, book_hash


def test_sota_state_roundtrip(tmp_path):
    s = SOTAState()
    s.set_book([{"factor_id": "a", "code": "x = 1"}])
    s.sota_executor = {"executor_id": "e1", "code": "y", "genome_id": "g1"}
    s.frozen_signals_version = 2
    s.frozen_signals_manifest = "/some/manifest.json"
    s.block_index = 3
    s.save(tmp_path / "joint_state.json")

    r = SOTAState.load(tmp_path / "joint_state.json")
    assert r.to_dict() == s.to_dict()
    assert r.book_hash == book_hash(s.book) != ""


def test_book_hash_is_whitespace_insensitive_and_code_sensitive():
    a = [{"factor_id": "f", "code": "x = 1\ny = 2"}]
    b = [{"factor_id": "f", "code": "x=1\n  y  =  2"}]
    c = [{"factor_id": "f", "code": "x = 3"}]
    assert book_hash(a) == book_hash(b)
    assert book_hash(a) != book_hash(c)


def test_ledger_family_vs_joint_billing():
    led = TrialsLedger()
    led.bill("factor", 5)                    # 5 new factor hypotheses
    led.bill("exec", 3)                      # 3 new executor hypotheses
    led.bill("exec", 4, rescore=True)        # archive re-score: looks only
    led.bill_look(2, source="joint_objective")

    assert led.family_count("factor") == 5
    assert led.family_count("exec") == 3     # UNCHANGED by the re-score
    assert led.joint_count() == 5 + 3 + 4 + 2
    assert led.joint_count() >= led.n_factor + led.n_exec  # the invariant

    r = TrialsLedger.from_dict(led.to_dict())
    assert (r.n_factor, r.n_exec, r.n_joint_looks) == (5, 3, 14)


def test_ledger_rejects_bad_input():
    led = TrialsLedger()
    with pytest.raises(ValueError):
        led.bill("model")                    # no third arm (locked decision 4)
    with pytest.raises(ValueError):
        led.bill("factor", -1)
    led.bill("factor", 0)                    # no-op, not an error
    assert led.joint_count() == 0
