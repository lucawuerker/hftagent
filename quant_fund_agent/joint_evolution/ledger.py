"""The shared cross-arm trials ledger (J0) — the joint run's honesty layer.

Two statistically distinct counts (``docs/joint-evolution/DESIGN.md`` §Ledger):

* **Per-family counts** (``n_factor``, ``n_exec``) — unique candidates *scored*
  per arm across every block.  They drive each arm's own within-search
  deflation gates: the ``√(2·ln N)``-type haircut corrects the within-family
  maximum, and factor ICs vs executor Sharpes are different test families with
  different nulls — cross-billing would be statistically wrong AND would break
  byte-identity of the standalone baselines.
* **Joint look count** (``n_joint_looks``) — every harness evaluation that
  reads VAL, across both arms, **including archive re-scores after a
  re-freeze** (a re-score is not a new hypothesis, but it IS a fresh look at
  VAL — one more draw in the implicit max the run performs).  By construction
  ``n_joint_looks ≥ n_factor + n_exec``.  The final publish filter and the
  touch-once TEST pass deflate by this count.

Scheduler decisions are deliberately NOT billed — the scheduler reallocates
which looks happen, it does not add hypotheses; its adaptivity is priced by
the joint walk-forward instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ARMS = ("factor", "exec")


@dataclass
class TrialsLedger:
    n_factor: int = 0
    n_exec: int = 0
    n_joint_looks: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def bill(self, arm: str, n: int = 1, *, rescore: bool = False) -> None:
        """Bill ``n`` evaluations of ``arm``.

        ``rescore=True`` = a deterministic re-evaluation of an already-counted
        hypothesis against new frozen signals: bills the joint look count only.
        """
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r} (use one of {ARMS})")
        n = int(n)
        if n < 0:
            raise ValueError(f"cannot bill a negative count ({n})")
        if n == 0:
            return
        if not rescore:
            if arm == "factor":
                self.n_factor += n
            else:
                self.n_exec += n
        self.n_joint_looks += n
        self.history.append({"arm": arm, "n": n, "rescore": bool(rescore)})

    def bill_look(self, n: int = 1, source: str = "joint_objective") -> None:
        """Bill VAL looks that belong to neither hypothesis family.

        Used for the block-boundary joint-objective scores and any other
        evaluation that reads VAL without proposing a new candidate.
        """
        n = int(n)
        if n <= 0:
            return
        self.n_joint_looks += n
        self.history.append({"arm": source, "n": n, "rescore": True})

    def family_count(self, arm: str) -> int:
        """The arm's own deflation-gate trial count."""
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r} (use one of {ARMS})")
        return self.n_factor if arm == "factor" else self.n_exec

    def joint_count(self) -> int:
        """The conservative count for publish / touch-once TEST deflation."""
        return self.n_joint_looks

    def to_dict(self) -> dict[str, Any]:
        return {"n_factor": self.n_factor, "n_exec": self.n_exec,
                "n_joint_looks": self.n_joint_looks, "history": self.history}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrialsLedger":
        return cls(n_factor=int(d.get("n_factor", 0)),
                   n_exec=int(d.get("n_exec", 0)),
                   n_joint_looks=int(d.get("n_joint_looks", 0)),
                   history=list(d.get("history", [])))
