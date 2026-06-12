"""Track 3 — the full downstream agentic fund on each prerun's factors.

Point the agents at one prerun's factor set (compose its DB, set
``FACTOR_DB_PATH``), build ``n_strategies`` strategies through
Selector → Architect → Statistician (each gets a held-out **OOS** verdict), keep
the approved ones, then run the Portfolio Manager to allocate.  The reported
numbers — per-strategy OOS Sharpe / IC / deflated-Sharpe and the PM's expected
portfolio metrics — mirror exactly what ``run_fund.py`` surfaces.

This is the only LLM-spending track.  Each prerun gets its own isolated book under
``<output_dir>/downstream/<prerun>/`` so comparisons never clobber each other.
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("comparison.downstream")

# Committee personalities (mirrors run_fund.py's default fund committee).
_DEFAULT_PERSONALITIES = ("defensive", "balanced", "aggressive")


def evaluate_prerun_downstream(prerun: str, cfg) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build K strategies + PM-allocate on one prerun; return (summary, per-strategy)."""
    from quant_fund_agent import pipeline
    from quant_fund_agent.databases import PortfolioDatabase, StrategyDatabase
    from quant_fund_agent.factors import preruns
    from quant_fund_agent.schemas import PMMode, PMPersonality

    out_dir = cfg.output_dir / "downstream" / prerun
    composed = out_dir / "factor_db.json"
    n_factors = preruns.build_downstream_factor_db(
        composed, prerun, include_seeds=cfg.include_seeds)

    # Scope the agents to this prerun's factors + the chosen data/universe.
    os.environ["FACTOR_DB_PATH"] = str(composed)
    os.environ["DATA_DIR"] = cfg.data_dir
    if cfg.n_tickers is not None:
        os.environ["ARCHITECT_N_TICKERS"] = str(cfg.n_tickers)

    strat_db = StrategyDatabase(returns_dir=out_dir / "returns")
    port_db = PortfolioDatabase()

    strat_rows: list[dict[str, Any]] = []
    n_approved = 0
    for k in range(cfg.n_strategies):
        try:
            res = pipeline.run_strategy_pipeline(
                oos_ratio=cfg.oos_split_ratio,
                target_horizon=cfg.target_horizon,
                run_statistician=True,
            )
        except Exception as e:  # noqa: BLE001 — one strategy must not abort the prerun
            log.warning("downstream %s strategy %d failed: %s", prerun, k, e)
            strat_rows.append({"prerun": prerun, "idx": k, "approved": False,
                               "reject_stage": "error", "error": str(e)[:200]})
            continue

        row: dict[str, Any] = {
            "prerun": prerun, "idx": k,
            "approved": bool(res.approved), "reject_stage": res.reject_stage,
        }
        if res.approved and res.strategy_spec is not None:
            record = pipeline.persist_approved_strategy(strat_db, res)
            if record is not None:
                n_approved += 1
                m, oos = record.backtest_metrics, record.oos_backtest_metrics
                row.update({
                    "strategy_id": record.id,
                    "model_type": record.model_type,
                    "n_factors": len(record.factor_ids),
                    "is_sharpe": getattr(m, "sharpe_ratio", None),
                    "oos_sharpe": getattr(oos, "sharpe_ratio", None),
                    "oos_ic": getattr(oos, "ic_mean", None),
                    "oos_ann_return": getattr(oos, "annualised_return", None),
                    "deflated_sharpe": (record.metadata or {}).get("deflated_sharpe_ratio"),
                })
        strat_rows.append(row)

    summary: dict[str, Any] = {
        "prerun": prerun, "n_factors": n_factors,
        "n_attempted": cfg.n_strategies, "n_approved": n_approved,
    }

    if n_approved > 0:
        personalities = [PMPersonality(p) for p in _DEFAULT_PERSONALITIES]
        port = pipeline.run_pm_rebalance(
            strat_db, port_db,
            personalities=personalities if cfg.committee else None,
            mode=PMMode.SELECTOR, committee=cfg.committee,
            pm_name=f"cmp_{prerun}",
        )
        if port is not None:
            em = port.expected_metrics or {}
            summary.update({
                "portfolio_expected_sharpe": em.get("expected_sharpe"),
                "portfolio_expected_vol": em.get("expected_volatility"),
                "portfolio_expected_return": em.get("expected_return"),
                "n_allocations": sum(1 for a in port.allocations if a.enabled),
            })

    approved = [r for r in strat_rows if r.get("approved")]
    oos = [r["oos_sharpe"] for r in approved if r.get("oos_sharpe") is not None]
    if oos:
        import numpy as np
        summary["mean_oos_sharpe"] = float(np.mean(oos))
        summary["median_oos_sharpe"] = float(np.median(oos))

    return summary, strat_rows
