"""J4: the joint walk-forward — the thesis's headline validation protocol.

For each fold boundary ``d_i`` the WHOLE outer loop — scheduler, blocks,
re-freezes, both arms — re-runs from scratch with ``cutoff_date=d_i`` (so no
candidate, freeze, or scheduler decision ever sees data past the boundary),
then the final ``(book, executor)`` pair is scored **once** on
``[d_i, d_{i+1})`` via the touch-once ``score_joint_oos``.  Inside a fold, VAL
re-burning is a search-efficiency concern, not a validity concern: the
reported number never saw the scoring window.

Validation only: every fold gets its own ``fold_<i>/`` directory with a fresh
ledger and scheduler posterior; nothing is persisted to factor DBs.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from quant_fund_agent.joint_evolution.loop import JointEvolutionLoop, JointRunConfig

log = logging.getLogger("joint_evolution.walkforward")


def run_joint_walk_forward(
    cfg: JointRunConfig,
    factor_cfg: Any,
    exec_cfg: Any,
    boundaries: Sequence[str],
    *,
    out_dir: str | Path,
    data_context: str = "",
    fields: list[str] | None = None,
    initial_factor_programs: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Re-run the whole joint loop per fold; touch-once score each archive.

    ``boundaries`` = ISO dates ``d_0 < d_1 < …``; fold *i* searches with
    ``cutoff_date=d_i`` and is scored on ``[d_i, d_{i+1})`` (the last fold
    scores on ``[d_last, ∞)``).  Returns (and writes) ``walkforward.json``.
    """
    from quant_fund_agent.mcp import research_client

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    boundaries = list(boundaries)
    if len(boundaries) < 1:
        raise ValueError("need at least one fold boundary date")

    folds: list[dict[str, Any]] = []
    for i, d_i in enumerate(boundaries):
        d_next = boundaries[i + 1] if i + 1 < len(boundaries) else None
        fold_dir = out_dir / f"fold_{i}"
        log.info("── joint walk-forward fold %d/%d: cutoff=%s, score on [%s, %s) ──",
                 i + 1, len(boundaries), d_i, d_i, d_next or "∞")

        fold_cfg = replace(cfg, out_dir=str(fold_dir), cutoff_date=d_i)
        fold_factor = replace(factor_cfg, cutoff_date=d_i,
                              out_dir=str(fold_dir / "factor"))
        fold_exec = replace(exec_cfg, cutoff_date=d_i,
                            out_dir=str(fold_dir / "exec"))
        t0 = time.time()
        loop = JointEvolutionLoop(fold_cfg, fold_factor, fold_exec,
                                  data_context=data_context, fields=fields)
        summary = loop.run(initial_factor_programs=initial_factor_programs)

        score: dict[str, Any] = {"ok": False, "error": "no book to score"}
        if loop.sota.book:
            score = research_client.score_joint_oos(
                loop.sota.book, loop.sota.sota_executor,
                start=d_i, end=d_next,
                target_horizon=cfg.target_horizon,
                data_dir=cfg.data_dir, n_tickers=cfg.n_tickers,
                fields=cfg.fields, model=cfg.joint_model,
                cost_rate=cfg.cost_rate)
        folds.append({
            "fold": i, "cutoff": d_i, "score_start": d_i, "score_end": d_next,
            "search_summary": summary,
            "oos": score,
            "elapsed_sec": round(time.time() - t0, 1),
        })
        (out_dir / "walkforward.json").write_text(
            json.dumps({"folds": folds}, indent=2, default=str))

    oks = [f["oos"] for f in folds if f["oos"].get("ok")]
    result = {
        "n_folds": len(folds),
        "n_scored": len(oks),
        "mean_oos_net_sharpe": (
            sum(f["oos_net_sharpe"] for f in oks if f["oos_net_sharpe"] is not None)
            / max(1, sum(1 for f in oks if f["oos_net_sharpe"] is not None))
            if oks else None),
        "folds": folds,
    }
    (out_dir / "walkforward.json").write_text(
        json.dumps(result, indent=2, default=str))
    return result
