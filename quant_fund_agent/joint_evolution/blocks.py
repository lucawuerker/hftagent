"""Running one arm for one block (J0).

A **block** = one incremental evolution session of one arm for
``gens_per_block`` generations.  The arm loops are unchanged code paths —
blocks are just (re)constructed loops driven through the E1 block API
(``run(resume=…, n_generations=…)``), so the sequential joint run is
byte-identical to standalone runs (the key J0 regression test).

Family-count accounting: the arm's own ``controller.n_trials`` *is* its family
count (each arm owns its out_dir), so after a block the ledger is billed with
the delta — no double bookkeeping, no injection.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

from quant_fund_agent.joint_evolution.ledger import TrialsLedger

log = logging.getLogger("joint_evolution.blocks")


@dataclass
class BlockResult:
    arm: str
    block_index: int
    generations: int
    n_trials_before: int
    n_trials_after: int
    archive_size: int
    elapsed_sec: float
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def candidates_scored(self) -> int:
        return self.n_trials_after - self.n_trials_before

    def to_dict(self) -> dict[str, Any]:
        return {"arm": self.arm, "block": self.block_index,
                "generations": self.generations,
                "candidates_scored": self.candidates_scored,
                "n_trials_after": self.n_trials_after,
                "archive_size": self.archive_size,
                "elapsed_sec": round(self.elapsed_sec, 1)}


def _prior_state_exists(out_dir: str | Path) -> bool:
    return (Path(out_dir) / "state.json").exists()


def run_factor_block(
    factor_cfg: Any,
    *,
    block_index: int,
    gens: int,
    ledger: TrialsLedger,
    initial_programs: Sequence[Any] | None = None,
    data_context: str = "",
    fields: list[str] | None = None,
    sota_executor: dict[str, Any] | None = None,
) -> tuple[BlockResult, Any]:
    """One factor-arm block.  Returns (result, the live EvolutionLoop)."""
    from quant_fund_agent.agents.factor_research.evolution.loop import EvolutionLoop

    resume = _prior_state_exists(factor_cfg.out_dir)
    if sota_executor is not None and hasattr(factor_cfg, "sota_executor"):
        factor_cfg = replace(factor_cfg, sota_executor=sota_executor)  # J3 coupling
    loop = EvolutionLoop(factor_cfg, data_context=data_context,
                         fields=fields or ["open", "high", "low", "close", "volume"])
    before = None
    t0 = time.time()
    if resume:
        summary = loop.run(resume=True, n_generations=gens)
    else:
        summary = loop.run(initial_programs=initial_programs, n_generations=gens)
    after = loop.controller.n_trials
    before = after - _billable(summary, loop, resume)
    result = BlockResult(arm="factor", block_index=block_index, generations=gens,
                         n_trials_before=before, n_trials_after=after,
                         archive_size=len(loop.controller.archive),
                         elapsed_sec=time.time() - t0, summary=summary)
    ledger.bill("factor", result.candidates_scored)
    return result, loop


def run_exec_block(
    exec_cfg: Any,
    *,
    block_index: int,
    gens: int,
    ledger: TrialsLedger,
    signals_manifest: str,
) -> tuple[BlockResult, Any]:
    """One exec-arm block.  Returns (result, the live ExecEvolutionLoop)."""
    from quant_fund_agent.agents.execution_research.evolution.loop import (
        ExecEvolutionLoop,
    )

    exec_cfg = replace(exec_cfg, signals_manifest=signals_manifest)
    resume = _prior_state_exists(exec_cfg.out_dir)
    loop = ExecEvolutionLoop(exec_cfg)
    t0 = time.time()
    summary = loop.run(resume=resume, n_generations=gens)
    after = loop.controller.n_trials
    before = after - _billable(summary, loop, resume)
    result = BlockResult(arm="exec", block_index=block_index, generations=gens,
                         n_trials_before=before, n_trials_after=after,
                         archive_size=len(loop.controller.archive),
                         elapsed_sec=time.time() - t0, summary=summary)
    ledger.bill("exec", result.candidates_scored)
    return result, loop


def _billable(summary: dict[str, Any], loop: Any, resumed: bool) -> int:
    """Candidates scored IN THIS BLOCK = n_trials_after − n_trials at entry.

    On a fresh run the loop started at 0; on a resumed run the checkpoint's
    counter was reloaded, so the block's own contribution is the difference to
    the persisted value at entry — which the loop recorded before running.
    """
    entry = getattr(loop, "_n_trials_at_entry", None)
    if entry is not None:
        return loop.controller.n_trials - entry
    # fallback: fresh run → everything is this block's
    return loop.controller.n_trials if not resumed else 0
