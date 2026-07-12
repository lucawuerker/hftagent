"""Tests for the block schedulers (J1)."""

from __future__ import annotations

import pytest

from quant_fund_agent.joint_evolution.scheduler import (
    RandomScheduler,
    RoundRobinScheduler,
    SequentialScheduler,
    make_scheduler,
)


def test_sequential_is_the_two_stage_plan():
    s = SequentialScheduler(total_blocks=5)          # default split: 3 factor
    assert [s.pick(b, []) for b in range(5)] == \
        ["factor", "factor", "factor", "exec", "exec"]
    s2 = SequentialScheduler(total_blocks=4, n_factor_blocks=1)
    assert [s2.pick(b, []) for b in range(4)] == \
        ["factor", "exec", "exec", "exec"]


def test_round_robin_alternates_factor_first():
    s = RoundRobinScheduler()
    assert [s.pick(b, []) for b in range(4)] == \
        ["factor", "exec", "factor", "exec"]


def test_random_is_seeded_and_forces_factor_first():
    a = [RandomScheduler(seed=7).pick(b, []) for b in range(8)]
    b = [RandomScheduler(seed=7).pick(k, []) for k in range(8)]
    c = [RandomScheduler(seed=8).pick(k, []) for k in range(8)]
    assert a == b                                 # deterministic per seed
    assert a[0] == "factor"                       # cold-start guarantee
    assert {x for x in a} <= {"factor", "exec"}
    assert a != c or len(set(a)) == 1             # different seed → (almost surely) different


def test_make_scheduler_dispatch():
    assert make_scheduler("sequential", total_blocks=4).kind == "sequential"
    assert make_scheduler("round_robin", total_blocks=4).kind == "round_robin"
    assert make_scheduler("random", total_blocks=4, seed=1).kind == "random"
    with pytest.raises(ValueError, match="unknown scheduler"):
        make_scheduler("llm_vibes", total_blocks=4)


# ── J2: the contextual Thompson-sampling bandit ────────────────────────────────

def _simulate(bandit, n, reward_of, history=None):
    """Drive the bandit like the outer loop does; returns the arm choices."""
    history = history if history is not None else []
    choices = []
    for b in range(len(history), len(history) + n):
        arm = "factor" if b == 0 else bandit.pick(b, history)
        r = reward_of(arm)
        bandit.update(arm, r)
        history.append({"arm": arm, "J_after": r, "reward": r,
                        "joint_score": {"val_net_sharpe": r},
                        "frozen_signals_version": 1})
        choices.append(arm)
    return choices


def test_bandit_warmup_forces_one_observation_per_arm():
    from quant_fund_agent.joint_evolution.bandit import BanditScheduler

    b = BanditScheduler(seed=3)
    choices = _simulate(b, 4, lambda arm: 0.0)
    assert set(choices[:2]) == {"factor", "exec"}   # both arms observed early


def test_bandit_converges_to_the_better_arm():
    from quant_fund_agent.joint_evolution.bandit import BanditScheduler

    for context in ("on", "off"):
        b = BanditScheduler(seed=5, context=context)
        choices = _simulate(
            b, 14, lambda arm: 0.10 if arm == "factor" else -0.10)
        tail = choices[-6:]
        assert tail.count("factor") >= 5, (context, choices)


def test_bandit_is_deterministic_per_seed_and_roundtrips():
    from quant_fund_agent.joint_evolution.bandit import BanditScheduler

    a = _simulate(BanditScheduler(seed=9), 10,
                  lambda arm: 0.05 if arm == "exec" else 0.0)
    b = _simulate(BanditScheduler(seed=9), 10,
                  lambda arm: 0.05 if arm == "exec" else 0.0)
    assert a == b

    # posterior save/load: a restored bandit continues the identical schedule
    src = BanditScheduler(seed=9)
    hist = []
    _simulate(src, 6, lambda arm: 0.05 if arm == "exec" else 0.0, hist)
    clone = BanditScheduler(seed=0)
    clone.load_state(src.state_dict())
    a_next = _simulate(src, 4, lambda arm: 0.05 if arm == "exec" else 0.0,
                       list(hist))
    b_next = _simulate(clone, 4, lambda arm: 0.05 if arm == "exec" else 0.0,
                       list(hist))
    assert a_next == b_next


def test_make_scheduler_builds_the_bandit():
    from quant_fund_agent.joint_evolution.bandit import BanditScheduler
    s = make_scheduler("bandit", total_blocks=6, seed=2)
    assert isinstance(s, BanditScheduler)


def test_context_vector_is_deterministic_and_bounded():
    from quant_fund_agent.joint_evolution.bandit import DIM, context_vector

    assert list(context_vector([])) == [1.0] + [0.0] * (DIM - 1)
    hist = [{"J_after": 0.02, "frozen_signals_version": 2,
             "joint_score": {"val_net_sharpe": 0.03, "val_gross_sharpe": 0.04,
                             "dsr_prob": 0.6, "mean_turnover": 0.4,
                             "n_book": 5}}]
    x = context_vector(hist)
    assert x[0] == 1.0 and abs(x[1] - 0.02) < 1e-12 and len(x) == DIM
    assert (context_vector(hist) == x).all()
