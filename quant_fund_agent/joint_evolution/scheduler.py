"""Block schedulers for the joint outer layer (J1; the bandit lands in J2).

All deterministic or classical-stochastic — an LLM-chosen schedule would let
language output steer which hypotheses get tested against VAL (RD-Agent(Q)
ablates one and it loses to the bandit anyway).  Common contract:

* ``pick(block_index, history) -> "factor" | "exec"``
* ``update(arm, reward, context)`` — a no-op except for the bandit.
* Block 0 is ALWAYS ``factor`` (the exec arm needs a book to freeze evaluation
  signals from), enforced by the loop, not the scheduler.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Scheduler:
    kind = "base"

    def pick(self, block_index: int, history: list[dict[str, Any]]) -> str:
        raise NotImplementedError

    def update(self, arm: str, reward: float | None,
               context: Any | None = None) -> None:
        return None

    def state_dict(self) -> dict[str, Any]:
        return {"kind": self.kind}

    def load_state(self, d: dict[str, Any]) -> None:
        return None


class SequentialScheduler(Scheduler):
    """All factor blocks, then all exec blocks — the original two-stage plan.

    ``n_factor_blocks`` defaults to half the total (rounded up).
    """

    kind = "sequential"

    def __init__(self, total_blocks: int, n_factor_blocks: int | None = None):
        self.total_blocks = int(total_blocks)
        self.n_factor_blocks = (int(n_factor_blocks) if n_factor_blocks is not None
                                else (self.total_blocks + 1) // 2)

    def pick(self, block_index: int, history: list[dict[str, Any]]) -> str:
        return "factor" if block_index < self.n_factor_blocks else "exec"


class RoundRobinScheduler(Scheduler):
    """Deterministic alternation, factor first — the default (zero tuning)."""

    kind = "round_robin"

    def pick(self, block_index: int, history: list[dict[str, Any]]) -> str:
        return "factor" if block_index % 2 == 0 else "exec"


class RandomScheduler(Scheduler):
    """Uniform coin flip per block (seeded) — RD-Agent(Q)'s ablation control."""

    kind = "random"

    def __init__(self, seed: int = 0):
        self.seed = int(seed)

    def pick(self, block_index: int, history: list[dict[str, Any]]) -> str:
        if block_index == 0:
            return "factor"
        # seeded per (seed, block) so a resumed run repeats the same schedule
        rng = np.random.default_rng(self.seed * 100_003 + block_index)
        return "factor" if rng.random() < 0.5 else "exec"


def make_scheduler(kind: str, *, total_blocks: int, seed: int = 0,
                   n_factor_blocks: int | None = None,
                   bandit_context: str = "on") -> Scheduler:
    if kind == "sequential":
        return SequentialScheduler(total_blocks, n_factor_blocks)
    if kind == "round_robin":
        return RoundRobinScheduler()
    if kind == "random":
        return RandomScheduler(seed)
    if kind == "bandit":
        from quant_fund_agent.joint_evolution.bandit import BanditScheduler
        return BanditScheduler(seed=seed, context=bandit_context)
    raise ValueError(f"unknown scheduler kind {kind!r} "
                     "(sequential | round_robin | random | bandit)")
