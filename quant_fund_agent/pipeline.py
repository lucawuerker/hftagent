"""Reusable fund-pipeline orchestration.

This module is the single, importable place where the *sequence* of agent
stages lives, so the notebook, the end-to-end script (``run_fund.py``),
the LangGraph orchestrator, and the future backtest all drive the agents
the same way.

Stages
------
1. ``run_research_session``    – (optional, weekly) Factor Researcher
   invents new alpha factors from papers and persists the good ones.
2. ``run_strategy_pipeline``   – Selector → Architect → Statistician.
   Produces one candidate strategy and an accept/reject verdict.
3. ``persist_approved_strategy`` – turn an *accepted* candidate into a
   ``StrategyRecord`` (in-sample + out-of-sample metrics) and register it
   — together with its per-bar PnL series — into the ``StrategyDatabase``.
   **This is the step that was missing before: without it the strategy
   book stays empty and the Portfolio Manager is a no-op.**
4. ``run_pm_rebalance``        – run one PM (or a committee of PMs) over
   the current book and write a ``PortfolioRecord``.

Why a module and not just a script
----------------------------------
The next milestone is a two-month, 10-second-bar backtest in which the
fund "trades" daily and the research agents re-run weekly.  That loop is

    for day in trading_days:
        append_live_returns(...)                       # mark-to-market
        if is_week_boundary(day):
            run_research_session(session_id=week_tag, cutoff_date=day)
            res = run_strategy_pipeline(cutoff_date=day)
            persist_approved_strategy(strategy_db, res)
            run_pm_rebalance(strategy_db, portfolio_db, ...)

i.e. exactly the functions below, called on a schedule.  Keeping them
here (rather than inlined in a script) means the backtest reuses the
identical code path the check-in demo uses.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from quant_fund_agent.databases import PortfolioDatabase, StrategyDatabase
from quant_fund_agent.portfolio import get_personality_profile
from quant_fund_agent.schemas import (
    ConstructionMethod,
    PMMode,
    PMPersonality,
    PortfolioRecord,
    StrategyBacktestMetrics,
    StrategyRecord,
    StrategyStatus,
    VotingMethod,
)

log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Canonical on-disk locations (single source of truth for every caller)
# ---------------------------------------------------------------------------

FACTOR_DB_PATH = Path("data/factors/factor_db.json")
STRATEGY_DB_PATH = Path("data/strategies/strategy_db.json")
PORTFOLIO_DB_PATH = Path("data/portfolio/portfolio_db.json")

OOS_TEST_ID = "out_of_sample_backtest"

# 10-second bars: 6.5h × 60min × 6 bars/min × 252 trading days.  This is the
# correct covariance annualisation factor for the fund's intraday returns, so
# that per-bar covariance lines up with the annualised returns stored on each
# StrategyRecord (otherwise expected-Sharpe figures blow up by ~sqrt(BPY)).
BARS_PER_YEAR = 2340 * 252


# ---------------------------------------------------------------------------
# Small serialisation helpers
# ---------------------------------------------------------------------------

def deserialise_series(payload: dict | None) -> pd.Series | None:
    """Inverse of the architect's ``_serialise_series`` helper."""
    if not payload or "timestamps" not in payload:
        return None
    idx = pd.to_datetime(payload["timestamps"])
    return pd.Series(payload["values"], index=idx, name="portfolio_return")


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (name or "strategy").lower()).strip("_")
    return slug or "strategy"


def _metrics_from_dict(d: dict | None) -> StrategyBacktestMetrics | None:
    """Validate a metrics dict into ``StrategyBacktestMetrics``.

    The architect's metrics dict carries a few extra keys (``equity_curve``,
    ``portfolio_returns``, ``backtest_elapsed_s``) that aren't model fields;
    Pydantic ignores them, which is exactly what we want.
    """
    if not d:
        return None
    return StrategyBacktestMetrics.model_validate(d)


# ---------------------------------------------------------------------------
# Stage 1 — Factor Researcher (optional, weekly)
# ---------------------------------------------------------------------------

def run_research_session(
    session_id: str | None = None,
    cutoff_date: date | None = None,
    n_papers: int = 2,
    n_ideas: int = 3,
    ic_threshold: float = 0.01,
    horizon: int = 6,
    n_tickers: int | None = 15,
    reset: bool = False,
) -> dict[str, Any]:
    """Run one Factor Researcher session and persist any kept factors.

    Returns the raw graph output (``selected_papers``, ``factor_ideas``,
    ``candidates``, ``kept_factor_ids``, ``rejected_factor_ids``).
    """
    from quant_fund_agent.agents.factor_research.graph import (
        factor_research_graph,
        reset_researcher_state,
    )
    from quant_fund_agent.agents.factor_research.state import FactorResearcherState
    from quant_fund_agent.factors import discover_factors

    discover_factors()
    if reset:
        reset_researcher_state()

    state = FactorResearcherState(
        session_id=session_id or date.today().isoformat(),
        cutoff_date=cutoff_date,
        n_papers=n_papers,
        n_factor_ideas=n_ideas,
        ic_threshold=ic_threshold,
        ic_target_horizon=horizon,
        n_tickers=n_tickers,
    )
    log.info("Factor research session '%s' (papers=%d, ideas=%d, |IC@%d|>=%.3f)",
             state.session_id, n_papers, n_ideas, horizon, ic_threshold)
    return factor_research_graph.invoke(state)


# ---------------------------------------------------------------------------
# Stage 2 — Selector → Architect → Statistician
# ---------------------------------------------------------------------------

@dataclass
class StrategyPipelineResult:
    """Everything the strategy-research pipeline produced in one pass."""

    selector_result: dict[str, Any]
    architect_result: dict[str, Any]
    stat_result: dict[str, Any] | None
    approved: bool
    reject_stage: str | None = None  # "architect" | "statistician" | None

    # convenience views
    @property
    def hypothesis(self) -> str:
        return self.selector_result.get("hypothesis", "")

    @property
    def strategy_spec(self):
        return self.architect_result.get("strategy_spec")

    @property
    def is_metrics(self) -> dict[str, Any]:
        return self.architect_result.get("backtest_metrics", {})


def run_strategy_pipeline(
    max_iterations: int = 3,
    oos_ratio: float = 0.2,
    run_statistician: bool = True,
) -> StrategyPipelineResult:
    """Selector → Architect → (optionally) Statistician, in one call.

    The ``approved`` flag is True only when the architect approved *and*
    the statistician accepted (or the statistician was skipped on an
    architect approval).
    """
    from quant_fund_agent.agents.architect.graph import architect_graph
    from quant_fund_agent.agents.selector.graph import selector_graph
    from quant_fund_agent.agents.statistician.graph import statistician_graph

    selector_result = selector_graph.invoke({})
    log.info("Selector hypothesis: %s", selector_result.get("hypothesis", "")[:120])
    log.info("Selected factors: %s", selector_result.get("selected_factor_ids"))

    architect_result = architect_graph.invoke({
        "hypothesis": selector_result["hypothesis"],
        "selected_factor_ids": selector_result["selected_factor_ids"],
        "factor_catalog": selector_result["factor_catalog"],
        "max_iterations": max_iterations,
        "oos_split_ratio": oos_ratio,
    })


    if architect_result.get("decision") != "approve":
        log.info("Architect rejected the candidate — no strategy produced.")
        return StrategyPipelineResult(
            selector_result, architect_result, None,
            approved=False, reject_stage="architect",
        )

    if not run_statistician:
        return StrategyPipelineResult(
            selector_result, architect_result, None, approved=True,
        )

    stat_result = statistician_graph.invoke({
        "hypothesis": architect_result["hypothesis"],
        "strategy_spec": architect_result["strategy_spec"],
        "trial_history": architect_result["trial_history"],
        "backtest_metrics": architect_result["backtest_metrics"],
        "initial_reasoning": architect_result.get("initial_reasoning", ""),
        "final_reasoning": architect_result["strategy_spec"].reasoning,
        "oos_split_ratio": oos_ratio,
    })
    for r in stat_result["test_results"]:
        v = r.verdict.value.upper()
        val = f"{r.value:.4f}" if r.value is not None else "—"
        log.info(f"  [{v:4s}] {r.test_name:36s}  value={val}"
              + (f"  p={r.p_value:.4f}" if r.p_value is not None else ""))
        if r.summary:
            log.info(f"         {r.summary}")
    log.info(f"\n   Final verdict : {stat_result['final_decision'].upper()}")
    accepted = stat_result.get("final_decision") == "accept"
    #log.info("Statistician decision: %s", stat_result.get("final_decision"))
    return StrategyPipelineResult(
        selector_result, architect_result, stat_result,
        approved=accepted,
        reject_stage=None if accepted else "statistician",
    )

# ---------------------------------------------------------------------------
# Stage 3 — build + persist a StrategyRecord (the missing link)
# ---------------------------------------------------------------------------

def _oos_metrics_from_stat(stat_result: dict | None) -> StrategyBacktestMetrics | None:
    """Pull the OOS metrics the statistician's OOS test recorded."""
    if not stat_result:
        return None
    for r in stat_result.get("test_results", []):
        test_id = getattr(r, "test_id", None) or (r.get("test_id") if isinstance(r, dict) else None)
        if test_id != OOS_TEST_ID:
            continue
        details = getattr(r, "details", None) or (r.get("details") if isinstance(r, dict) else None)
        if details and "oos_metrics" in details:
            return _metrics_from_dict(details["oos_metrics"])
    return None


def build_strategy_record(
    result: StrategyPipelineResult,
    strategy_id: str | None = None,
) -> tuple[StrategyRecord, pd.Series | None]:
    """Turn an approved pipeline result into a ``StrategyRecord`` + returns.

    The returns series is the architect's *in-sample* per-bar PnL — enough
    to seed the cross-strategy correlation matrix the PM relies on.  In a
    live backtest this series is then extended bar-by-bar via
    ``StrategyDatabase.append_returns``.
    """
    spec = result.strategy_spec
    if spec is None:
        raise ValueError("build_strategy_record: result has no strategy_spec")

    sid = strategy_id or f"{_slugify(spec.strategy_name)}_{uuid.uuid4().hex[:8]}"

    is_metrics = _metrics_from_dict(result.is_metrics)
    oos_metrics = _oos_metrics_from_stat(result.stat_result)
    is_returns = deserialise_series(result.is_metrics.get("portfolio_returns"))

    deflated_sr = None
    final_reasoning = ""
    if result.stat_result:
        final_reasoning = result.stat_result.get("final_reasoning_statistician", "")
        for r in result.stat_result.get("test_results", []):
            tid = getattr(r, "test_id", "")
            if tid == "deflated_sharpe_ratio":
                deflated_sr = getattr(r, "value", None)

    model_type = spec.model_type or "static_weights"
    is_ml = model_type != "static_weights"
    factor_ids = list(spec.factor_ids) if spec.factor_ids else list(spec.weights.keys())

    record = StrategyRecord(
        id=sid,
        name=spec.strategy_name,
        class_name="ModelStrategy" if is_ml else "DynamicStrategy",
        description=result.hypothesis,
        factor_ids=factor_ids,
        status=StrategyStatus.APPROVED,
        backtest_metrics=is_metrics,
        oos_backtest_metrics=oos_metrics,
        model_type=model_type,
        model_params={
            "model_type": model_type,
            "model_hyperparams": dict(spec.model_params),
            "factor_ids": factor_ids,
            "target_horizon": spec.target_horizon,
            "model_artifact_path": spec.model_artifact_path,
            "weights": dict(spec.weights),
            "holding_period": spec.holding_period,
            "max_positions": spec.max_positions,
            "equal_weight": spec.equal_weight,
            "min_conviction": spec.min_conviction,
        },
        holding_period=spec.holding_period,
        metadata={
            "n_trials": len(result.architect_result.get("trial_history", [])),
            "deflated_sharpe_ratio": deflated_sr,
            "architect_reasoning": spec.reasoning,
            "statistician_reasoning": final_reasoning,
        },
    )
    return record, is_returns


def persist_approved_strategy(
    strategy_db: StrategyDatabase,
    result: StrategyPipelineResult,
    strategy_id: str | None = None,
) -> StrategyRecord | None:
    """Register an approved strategy (record + returns) into the book.

    Returns the persisted ``StrategyRecord`` or ``None`` if the pipeline
    did not produce an accepted strategy.
    """
    if not result.approved:
        return None
    record, returns = build_strategy_record(result, strategy_id=strategy_id)
    strategy_db.register_strategy(record, portfolio_returns=returns)
    log.info("Persisted strategy '%s' (%s) — IS Sharpe=%s, OOS Sharpe=%s",
             record.id, record.name,
             getattr(record.backtest_metrics, "sharpe_ratio", None),
             getattr(record.oos_backtest_metrics, "sharpe_ratio", None))
    return record


def strategy_from_record(record: StrategyRecord):
    """Rebuild a runnable ``DynamicStrategy`` from a persisted record.

    Forward-looking helper for the backtest: given a stored
    ``StrategyRecord``, reconstruct the exact strategy object so its
    signal can be recomputed on new (live) bars.
    """
    from quant_fund_agent.strategies.dynamic import DynamicStrategy

    p = record.model_params or {}
    model_type = record.model_type or p.get("model_type") or "static_weights"

    # Fitted ML strategy: reload the persisted estimator (never refit).
    if model_type != "static_weights":
        from quant_fund_agent.strategies.model_strategy import ModelStrategy

        return ModelStrategy.from_artifact(
            p.get("model_artifact_path", ""),
            strategy_id=record.id,
            name=record.name,
            description=record.description,
            holding_period=p.get("holding_period", record.holding_period),
            max_positions=p.get("max_positions", 20),
            equal_weight=p.get("equal_weight", False),
            min_conviction=p.get("min_conviction", 0.0),
        )

    return DynamicStrategy(
        strategy_id=record.id,
        name=record.name,
        description=record.description,
        factor_ids=list(record.factor_ids),
        weights=p.get("weights", {}),
        holding_period=p.get("holding_period", record.holding_period),
        max_positions=p.get("max_positions", 20),
        equal_weight=p.get("equal_weight", False),
        min_conviction=p.get("min_conviction", 0.0),
    )


# ---------------------------------------------------------------------------
# Stage 4 — Portfolio Manager rebalance (single PM or committee)
# ---------------------------------------------------------------------------

def run_pm_rebalance(
    strategy_db: StrategyDatabase,
    portfolio_db: PortfolioDatabase,
    personalities: list[PMPersonality] | None = None,
    mode: PMMode = PMMode.SELECTOR,
    committee: bool = False,
    voting_method: VotingMethod = VotingMethod.SIMPLE_AVERAGE,
    pm_name: str = "fund_pm",
    construction_method: ConstructionMethod | None = None,
    target_n_strategies: int | None = None,
    annualisation_factor: float = BARS_PER_YEAR,
) -> PortfolioRecord | None:
    """Run one PM (or a committee) over the current strategy book.

    * ``committee=False`` → a single PM (first personality, default
      BALANCED) writes a ``PortfolioRecord``.
    * ``committee=True``  → every listed personality proposes and the
      committee aggregates them into one consensus ``PortfolioRecord``.

    Returns the resulting ``PortfolioRecord`` (or ``None`` if the book is
    empty).
    """
    from quant_fund_agent.agents.portfolio_manager.committee import (
        CommitteeConfig,
        run_pm_committee,
    )
    from quant_fund_agent.agents.portfolio_manager.graph import (
        portfolio_manager_graph,
    )
    from quant_fund_agent.agents.portfolio_manager.state import PortfolioManagerState

    if not strategy_db.list_strategies():
        log.warning("run_pm_rebalance: strategy book is empty — nothing to allocate.")
        return None

    personalities = personalities or [PMPersonality.BALANCED]
    strategy_db.refresh_correlations()

    def _make_state(personality: PMPersonality, name: str) -> PortfolioManagerState:
        return PortfolioManagerState(
            pm_name=name,
            mode=mode,
            personality=personality,
            profile=get_personality_profile(personality),
            construction_method_override=construction_method,
            target_n_strategies=target_n_strategies,
            annualisation_factor=annualisation_factor,
            strategy_db=strategy_db,
            portfolio_db=portfolio_db,
        )

    if committee and len(personalities) > 1:
        states = [
            _make_state(p, f"{pm_name}_{p.value}") for p in personalities
        ]
        cfg = CommitteeConfig(voting_method=voting_method, pm_name=pm_name)
        return run_pm_committee(states, cfg, strategy_db, portfolio_db)

    state = _make_state(personalities[0], pm_name)
    result = portfolio_manager_graph.invoke(state)
    return result.get("portfolio_record")


# ---------------------------------------------------------------------------
# DB load / save convenience
# ---------------------------------------------------------------------------

def load_dbs(
    strategy_db_path: Path = STRATEGY_DB_PATH,
    portfolio_db_path: Path = PORTFOLIO_DB_PATH,
) -> tuple[StrategyDatabase, PortfolioDatabase]:
    """Load the persisted strategy + portfolio databases (empty if absent)."""
    sdb = StrategyDatabase()
    sdb.load_from_json(strategy_db_path)
    pdb = PortfolioDatabase()
    pdb.load_from_json(portfolio_db_path)
    return sdb, pdb


def save_dbs(
    strategy_db: StrategyDatabase,
    portfolio_db: PortfolioDatabase,
    strategy_db_path: Path = STRATEGY_DB_PATH,
    portfolio_db_path: Path = PORTFOLIO_DB_PATH,
) -> None:
    """Flush both databases to disk (including any cached returns series)."""
    strategy_db.flush_returns()
    strategy_db.save_to_json(strategy_db_path)
    portfolio_db.save_to_json(portfolio_db_path)
