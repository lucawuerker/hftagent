"""Run the strategy pipeline and dump EVERY attempt — approved and rejected.

``run_fund.py`` persists only approved strategies; the landing page's "Likely
overfit" card *is* a rejected candidate, so this driver dumps the full
``StrategyPipelineResult`` per attempt (spec, complete trial history including
each trial's serialized return series, complete stat-test details) to
``<scope>/landing_examples/candidates/attempt_<nn>.json``.

Approved strategies are still persisted into the scope's strategy DB exactly as
``run_fund.py`` would, so their artifacts and return CSVs land in the workspace.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("landing_examples.driver")

CANDIDATES_SUBDIR = Path("landing_examples") / "candidates"


def candidates_dir(scope) -> Path:
    return scope.dir / CANDIDATES_SUBDIR


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 — provenance is best-effort
        return None


def _next_attempt_index(cdir: Path) -> int:
    existing = sorted(cdir.glob("attempt_*.json"))
    if not existing:
        return 1
    last = existing[-1].stem.split("_")[-1]
    return int(last) + 1


def dump_candidate(result, attempt: int, strategy_id: str | None,
                   oos_ratio: float, universe: dict[str, Any]) -> dict[str, Any]:
    """Full JSON-able view of one pipeline attempt (approved or rejected)."""
    sel = result.selector_result or {}
    arch = result.architect_result or {}
    spec = result.strategy_spec

    stat = None
    if result.stat_result is not None:
        stat = {
            "final_decision": result.stat_result.get("final_decision"),
            "final_reasoning": result.stat_result.get(
                "final_reasoning_statistician", ""),
            "test_results": [
                r.model_dump(mode="json") if hasattr(r, "model_dump") else r
                for r in result.stat_result.get("test_results", [])
            ],
        }

    return {
        "attempt": attempt,
        "dumped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "approved": result.approved,
        "reject_stage": result.reject_stage,
        "strategy_id": strategy_id,
        "hypothesis": result.hypothesis,
        "selection_rationale": sel.get("selection_rationale", ""),
        "selected_factor_ids": list(sel.get("selected_factor_ids") or []),
        "spec": spec.model_dump(mode="json") if spec is not None else None,
        "backtest_metrics": arch.get("backtest_metrics", {}),
        "architect_decision": arch.get("decision"),
        "initial_reasoning": arch.get("initial_reasoning", ""),
        "trial_history": [
            t.model_dump(mode="json") if hasattr(t, "model_dump") else t
            for t in arch.get("trial_history", [])
        ],
        "stat_result": stat,
        "oos_split_ratio": oos_ratio,
        "universe": universe,
    }


def run_batch(
    config_name: str | None,
    prerun: str,
    n_strategies: int = 8,
    max_iterations: int = 6,
    oos_ratio: float = 0.2,
    no_seeds: bool = False,
    n_tickers: int = 0,
    target_horizon: int | None = None,
    position_construction: str = "auto",
) -> list[Path]:
    """Run ``n_strategies`` pipeline attempts under the scope, dumping each one.

    Mirrors ``run_fund.py``'s scope activation exactly; returns the dump paths.
    """
    from quant_fund_agent import pipeline
    from quant_fund_agent.config import default_config_name, get_settings
    from quant_fund_agent.factors import discover_factors
    from quant_fund_agent.workspace import Scope

    if n_tickers and n_tickers > 0:
        os.environ.setdefault("ARCHITECT_N_TICKERS", str(n_tickers))

    settings = get_settings()
    config_name = config_name or default_config_name(settings.data)
    scope = Scope(config_name, prerun)
    scope.ensure()
    scope.write_config_snapshot(settings.data)
    mismatch = scope.config_mismatch(settings.data)
    if mismatch:
        log.warning(mismatch)

    n_factors = scope.compose_runtime_db(include_seeds=not no_seeds)
    scope.activate(factor_db=scope.composed_db_path)
    log.info("Scope %s: %d factors visible", scope.label, n_factors)
    discover_factors()

    strategy_db, portfolio_db = pipeline.load_dbs(
        scope.strategy_db_path, scope.portfolio_db_path)

    data = settings.data
    universe = {
        "provider": data.provider,
        "asset_class": data.asset_class,
        "universe": data.universe_preset or ",".join(data.tickers or []),
        "frequency": data.frequency,
        "timespan": [data.start, data.end],
        "n_tickers_cap": os.getenv("ARCHITECT_N_TICKERS") or "all",
    }

    cdir = candidates_dir(scope)
    cdir.mkdir(parents=True, exist_ok=True)
    start_idx = _next_attempt_index(cdir)

    batch_meta = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "scope": scope.label,
        "prerun": prerun,
        "config_name": config_name,
        "n_strategies": n_strategies,
        "max_iterations": max_iterations,
        "oos_ratio": oos_ratio,
        "no_seeds": no_seeds,
        "universe": universe,
        "models": {
            "architect": os.getenv("ARCHITECT_LLM_MODEL", "gpt-4o-mini"),
            "selector": os.getenv("SELECTOR_LLM_MODEL", "gpt-4o-mini"),
            "statistician": os.getenv("STATISTICIAN_LLM_MODEL", "gpt-4o-mini"),
        },
        "env": {
            "ARCHITECT_N_TICKERS": os.getenv("ARCHITECT_N_TICKERS"),
            "QF_USE_MCP": os.getenv("QF_USE_MCP"),
        },
        "attempts": [],
    }

    dumped: list[Path] = []
    for i in range(n_strategies):
        attempt = start_idx + i
        log.info("--- attempt %d (%d/%d this batch) ---", attempt, i + 1, n_strategies)
        res = pipeline.run_strategy_pipeline(
            max_iterations=max_iterations,
            oos_ratio=oos_ratio,
            target_horizon=target_horizon,
            position_construction=position_construction,
        )
        strategy_id = None
        if res.approved:
            record = pipeline.persist_approved_strategy(strategy_db, res)
            strategy_id = record.id if record is not None else None

        payload = dump_candidate(res, attempt, strategy_id, oos_ratio, universe)
        path = cdir / f"attempt_{attempt:02d}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        dumped.append(path)
        batch_meta["attempts"].append({
            "attempt": attempt,
            "approved": res.approved,
            "reject_stage": res.reject_stage,
            "strategy_id": strategy_id,
        })
        log.info("attempt %d → %s (%s)", attempt,
                 "APPROVED" if res.approved else f"rejected@{res.reject_stage}",
                 path.name)

    pipeline.save_dbs(strategy_db, portfolio_db,
                      scope.strategy_db_path, scope.portfolio_db_path)

    batch_meta["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta_path = cdir / f"batch_{start_idx:02d}_meta.json"
    meta_path.write_text(json.dumps(batch_meta, indent=2, default=str))
    log.info("Dumped %d attempts → %s", len(dumped), cdir)
    return dumped


def load_candidates(scope) -> list[dict[str, Any]]:
    """All dumped attempts under the scope, sorted by attempt number."""
    cdir = candidates_dir(scope)
    out: list[dict[str, Any]] = []
    for p in sorted(cdir.glob("attempt_*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception as e:  # noqa: BLE001 — one corrupt dump must not hide the rest
            log.warning("skipping unreadable %s (%s)", p.name, e)
    return out


def load_batch_meta(scope) -> dict[str, Any] | None:
    """The most recent batch metadata (env/models/universe), if any."""
    metas = sorted(candidates_dir(scope).glob("batch_*_meta.json"))
    if not metas:
        return None
    try:
        return json.loads(metas[-1].read_text())
    except Exception:  # noqa: BLE001
        return None
