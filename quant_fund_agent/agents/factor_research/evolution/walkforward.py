"""Walk-forward final validation: re-run the WHOLE evolutionary loop per period.

The thesis results pass (design §overfit protocol, decision 2): blocked validation
stability is the *development-time* robustness engine; for the results chapter the **entire loop**
is re-run period-by-period so every discovery is out-of-sample to the next
period.  For each boundary date ``d_i`` in the schedule:

1. a **fresh** :class:`~quant_fund_agent.agents.factor_research.evolution.loop.
   EvolutionLoop` runs with ``cutoff_date = d_i`` (it can only ever see bars
   before ``d_i`` — the same seam the deployed fund's ``cutoff_date`` uses);
2. its final Pareto archive is scored **once** on ``[d_i, d_{i+1})`` via
   :func:`~quant_fund_agent.mcp.research_client.score_book_oos` (combined-model
   OOS IC, per-factor OOS ICs, CSCV PBO).

This module is a *validation* driver: fold archives are checkpointed under
``<out_dir>/fold_<i>/`` but nothing is persisted to the prerun's factor DB —
the factors the thesis reports were never selected on the windows they are
scored on.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from quant_fund_agent.agents.factor_research.evolution.genome import FactorProgram
from quant_fund_agent.agents.factor_research.evolution.loop import (
    EvolutionLoop,
    EvolutionRunConfig,
)

log = logging.getLogger("evolution.walkforward")


def run_walk_forward(
    base_cfg: EvolutionRunConfig,
    boundaries: Sequence[str],
    *,
    out_dir: str | Path | None = None,
    initial_programs: Sequence[FactorProgram] | None = None,
) -> dict[str, Any]:
    """Run the loop once per fold and score each archive on the next period.

    ``boundaries`` is an ascending list of ISO dates ``d0 < d1 < … < dk``:
    fold ``i`` evolves with ``cutoff_date=d_i`` and is scored on
    ``[d_i, d_{i+1})`` — so ``k`` boundaries give ``k−1`` folds.  Each fold gets
    a decorrelated seed (``base_cfg.seed + i``).  ``initial_programs`` (tests /
    LLM-free reruns) seeds every fold with the same starting programs instead
    of a fresh brainstorm.

    Returns the aggregate report (also written to ``<out_dir>/walkforward.json``).
    """
    boundaries = list(boundaries)
    if len(boundaries) < 2:
        raise ValueError("need at least two boundary dates (one fold)")
    if sorted(boundaries) != boundaries:
        raise ValueError(f"boundaries must ascend, got {boundaries}")

    from quant_fund_agent.mcp import research_client

    out = Path(out_dir) if out_dir is not None else Path(base_cfg.out_dir) / "walkforward"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    folds: list[dict[str, Any]] = []
    for i, cutoff in enumerate(boundaries[:-1]):
        oos_end = boundaries[i + 1]
        cfg = replace(base_cfg, cutoff_date=cutoff, seed=base_cfg.seed + i,
                      out_dir=str(out / f"fold_{i}"))
        log.info("── walk-forward fold %d/%d: evolve < %s, score on [%s, %s) ──",
                 i + 1, len(boundaries) - 1, cutoff, cutoff, oos_end)
        loop = EvolutionLoop(cfg)
        summary = loop.run(initial_programs=initial_programs)

        book = [{"factor_id": fid, "code": code}
                for fid, code in loop.controller.archive_programs()]
        if book:
            score = research_client.score_book_oos(
                book, start=cutoff, end=oos_end,
                target_horizon=cfg.target_horizon, data_dir=cfg.data_dir,
                n_tickers=cfg.n_tickers, fields=loop.fields)
        else:
            score = {"ok": False, "error": "empty archive"}

        folds.append({
            "fold": i,
            "cutoff": cutoff,
            "oos_end": oos_end,
            "n_trials": summary["n_trials"],
            "archive_size": len(summary["archive"]),
            "archive": summary["archive"],
            "oos": score,
        })

    ics = [f["oos"].get("combined_oos_ic") for f in folds
           if f["oos"].get("ok") and f["oos"].get("combined_oos_ic") is not None]
    pbos = [f["oos"]["pbo"]["pbo"] for f in folds
            if f["oos"].get("ok") and f["oos"].get("pbo", {}).get("pbo") is not None]
    report = {
        "boundaries": boundaries,
        "n_folds": len(folds),
        "folds": folds,
        "mean_combined_oos_ic": (float(np.mean(ics)) if ics else None),
        "std_combined_oos_ic": (float(np.std(ics)) if len(ics) > 1 else None),
        "mean_pbo": (float(np.mean(pbos)) if pbos else None),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "walkforward.json").write_text(json.dumps(report, indent=2, default=str))
    log.info("walk-forward complete: mean combined OOS IC=%s over %d fold(s)",
             report["mean_combined_oos_ic"], len(folds))
    return report
