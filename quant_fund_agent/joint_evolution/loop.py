"""The joint outer loop (J0/J1): scheduler → block → re-freeze → score → repeat.

One joint run = one workspace prerun; everything lives under
``<scope>/joint/``::

    joint/joint_state.json      # SOTAState + TrialsLedger + scheduler + last_J
    joint/blocks.jsonl          # one row per block (the outer lineage)
    joint/frozen_signals/v<k>/  # the versioned interface artifacts
    joint/factor/               # the factor arm's out_dir (state/lineage/config)
    joint/exec/                 # the exec arm's out_dir

The loop checkpoints after **every block boundary**, so a joint run resumes
mid-schedule exactly like an arm run resumes mid-generation.  Block 0 is
always a factor block (the exec arm needs a book to freeze signals from).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

from quant_fund_agent.joint_evolution.blocks import (
    BlockResult,
    run_exec_block,
    run_factor_block,
)
from quant_fund_agent.joint_evolution.ledger import TrialsLedger
from quant_fund_agent.joint_evolution.objective import score_joint
from quant_fund_agent.joint_evolution.refreeze import (
    refreeze_after_factor_block,
    update_sota_executor,
)
from quant_fund_agent.joint_evolution.scheduler import make_scheduler
from quant_fund_agent.joint_evolution.state import SOTAState

log = logging.getLogger("joint_evolution.loop")


@dataclass
class JointRunConfig:
    """The outer layer's knobs (arm configs are passed alongside, not nested)."""

    out_dir: str = "data/joint_evolution"
    total_blocks: int = 4
    gens_per_block: int = 2
    scheduler: str = "round_robin"      # sequential | round_robin | random | bandit
    n_factor_blocks: int | None = None  # sequential split point (default: half)
    seed: int = 0
    coupling: bool = False              # J3: executor-aware factor cost gate
    bandit_context: str = "on"          # J2: "off" → non-contextual Gaussian TS
    # ── shared evaluation frame (identical for freeze + joint objective) ──
    target_horizon: int = 6
    is_frac: float = 0.6
    val_frac: float = 0.2
    cutoff_date: str | None = None
    data_dir: str = "ticker_data"
    n_tickers: int | None = 15
    fields: list[str] | None = None
    freeze_specs: list[dict[str, Any]] | None = None
    joint_model: str = "ridge"          # combiner for the joint objective J
    cost_rate: float = 5e-4
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JointEvolutionLoop:
    """Drive the block-coordinate alternation over the two arm loops."""

    def __init__(self, cfg: JointRunConfig, factor_cfg: Any, exec_cfg: Any,
                 *, data_context: str = "", fields: list[str] | None = None):
        self.cfg = cfg
        self.out_dir = Path(cfg.out_dir)
        self.factor_cfg = replace(factor_cfg, out_dir=str(self.out_dir / "factor"))
        self.exec_cfg = replace(exec_cfg, out_dir=str(self.out_dir / "exec"))
        self.data_context = data_context
        self.panel_fields = fields or ["open", "high", "low", "close", "volume"]
        self.sota = SOTAState()
        self.ledger = TrialsLedger()
        self.scheduler = make_scheduler(cfg.scheduler,
                                        total_blocks=cfg.total_blocks,
                                        seed=cfg.seed,
                                        n_factor_blocks=cfg.n_factor_blocks,
                                        bandit_context=cfg.bandit_context)
        self.last_J: float | None = None
        self.history: list[dict[str, Any]] = []
        self._factor_loop: Any | None = None
        self._exec_loop: Any | None = None

    # ── persistence ───────────────────────────────────────────────────────────

    @property
    def state_path(self) -> Path:
        return self.out_dir / "joint_state.json"

    def _checkpoint(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "sota": self.sota.to_dict(),
            "ledger": self.ledger.to_dict(),
            "scheduler": self.scheduler.state_dict(),
            "last_J": self.last_J,
            "run_config": self.cfg.to_dict(),
        }
        self.state_path.write_text(json.dumps(payload, indent=2, default=str))
        with (self.out_dir / "blocks.jsonl").open("w") as fh:
            for row in self.history:
                fh.write(json.dumps(row, default=str) + "\n")

    def _restore(self) -> bool:
        if not self.state_path.exists():
            return False
        payload = json.loads(self.state_path.read_text())
        self.sota = SOTAState.from_dict(payload["sota"])
        self.ledger = TrialsLedger.from_dict(payload["ledger"])
        self.scheduler.load_state(payload.get("scheduler", {}))
        self.last_J = payload.get("last_J")
        blocks_path = self.out_dir / "blocks.jsonl"
        if blocks_path.exists():
            self.history = [json.loads(line) for line in
                            blocks_path.read_text().splitlines() if line.strip()]
        log.info("resumed joint run at block %d (looks=%d, factor=%d, exec=%d)",
                 self.sota.block_index, self.ledger.n_joint_looks,
                 self.ledger.n_factor, self.ledger.n_exec)
        return True

    # ── one block ─────────────────────────────────────────────────────────────

    def _eval_kwargs(self) -> dict[str, Any]:
        return dict(target_horizon=self.cfg.target_horizon,
                    is_frac=self.cfg.is_frac, val_frac=self.cfg.val_frac,
                    cutoff_date=self.cfg.cutoff_date, data_dir=self.cfg.data_dir,
                    n_tickers=self.cfg.n_tickers, fields=self.cfg.fields)

    def _run_one_block(self, b: int, arm: str,
                       initial_factor_programs: Sequence[Any] | None,
                       ) -> tuple[BlockResult, dict[str, Any]]:
        boundary: dict[str, Any] = {}
        if arm == "factor":
            result, self._factor_loop = run_factor_block(
                self.factor_cfg, block_index=b, gens=self.cfg.gens_per_block,
                ledger=self.ledger,
                initial_programs=initial_factor_programs if b == 0 else None,
                data_context=self.data_context, fields=self.panel_fields,
                sota_executor=(self.sota.sota_executor
                               if self.cfg.coupling else None))
            boundary = refreeze_after_factor_block(
                self.sota, self._factor_loop.controller, self._exec_loop,
                self.ledger, out_dir=str(self.out_dir),
                specs=self.cfg.freeze_specs, **self._eval_kwargs())
        else:
            if not self.sota.frozen_signals_manifest:
                raise RuntimeError(
                    "exec block scheduled before any frozen signals exist — "
                    "block 0 must be a factor block")
            result, self._exec_loop = run_exec_block(
                self.exec_cfg, block_index=b, gens=self.cfg.gens_per_block,
                ledger=self.ledger,
                signals_manifest=self.sota.frozen_signals_manifest)
            boundary = update_sota_executor(self.sota, self._exec_loop)
        return result, boundary

    # ── the drive ─────────────────────────────────────────────────────────────

    def run(self, initial_factor_programs: Sequence[Any] | None = None,
            ) -> dict[str, Any]:
        t0 = time.time()
        self._restore()

        for b in range(self.sota.block_index, self.cfg.total_blocks):
            arm = "factor" if b == 0 else self.scheduler.pick(b, self.history)
            log.info("── block %d/%d: %s arm (%d gens) ──",
                     b + 1, self.cfg.total_blocks, arm, self.cfg.gens_per_block)
            result, boundary = self._run_one_block(b, arm,
                                                   initial_factor_programs)

            score = score_joint(self.sota, self.ledger,
                                model=self.cfg.joint_model,
                                cost_rate=self.cfg.cost_rate,
                                **self._eval_kwargs())
            J = score.get("J")
            reward = (J - self.last_J
                      if J is not None and self.last_J is not None else None)
            self.scheduler.update(arm, reward)

            row = {
                "block": b, "arm": arm, **result.to_dict(),
                "boundary": boundary,
                "J_before": self.last_J, "J_after": J, "reward": reward,
                "joint_score": {k: v for k, v in score.items()
                                if k not in ("ok",)},
                "ledger": {"n_factor": self.ledger.n_factor,
                           "n_exec": self.ledger.n_exec,
                           "n_joint_looks": self.ledger.n_joint_looks},
                "frozen_signals_version": self.sota.frozen_signals_version,
                "sota_executor": (self.sota.sota_executor or {}).get("executor_id"),
            }
            self.history.append(row)
            self.last_J = J if J is not None else self.last_J
            self.sota.block_index = b + 1
            self._checkpoint()
            log.info("block %d done: J=%s reward=%s ledger=%s",
                     b, J, reward, row["ledger"])

        return self.summary(time.time() - t0)

    def summary(self, elapsed: float) -> dict[str, Any]:
        return {
            "blocks": self.sota.block_index,
            "scheduler": self.cfg.scheduler,
            "J": self.last_J,
            "ledger": self.ledger.to_dict() | {"history": "..."},
            "book_size": len(self.sota.book),
            "frozen_signals_version": self.sota.frozen_signals_version,
            "sota_executor": (self.sota.sota_executor or {}).get("executor_id"),
            "arm_choices": [r["arm"] for r in self.history],
            "elapsed_sec": round(elapsed, 1),
        }
