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

from quant_fund_agent.data.frequency import (
    DEFAULT_BARS_PER_YEAR,
    periods_per_year_from_index,
)
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

# Legacy covariance annualisation factor (10-second bars: 6.5h × 360 bars/h ×
# 252 trading days).  Used as the fallback; ``run_pm_rebalance`` now infers the
# factor from the strategy returns' own frequency so per-bar covariance lines up
# with the annualised returns at any data frequency (daily → 252, 10-sec → this).
BARS_PER_YEAR = DEFAULT_BARS_PER_YEAR


def _infer_annualisation(strategy_db: StrategyDatabase) -> float:
    """Annualisation factor inferred from the strategy book's return frequency.

    Falls back to :data:`BARS_PER_YEAR` (legacy 10-sec assumption) when no
    returns are available to infer from.
    """
    for series in strategy_db.get_all_returns().values():
        if series is not None and len(series) > 2:
            return float(periods_per_year_from_index(series.index))
    return float(BARS_PER_YEAR)


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

def _infer_seconds_per_bar(data_dir: str) -> float | None:
    """Cheap probe of the feed's bar size (seconds per bar) for the prompts.

    Loads a *single* ticker's ``close`` column only, then takes the median
    spacing of its index (see :func:`data.frequency._median_bar_seconds`).  Used
    purely to tell the researcher how long a bar is so it can reason about
    prediction horizons in wall-clock time.  Best-effort: any failure (no local
    panel, API-only provider, …) returns ``None`` → feed-agnostic prompt wording.
    """
    try:
        from quant_fund_agent.backtesting.data_loader import load_panel
        from quant_fund_agent.data.frequency import _median_bar_seconds

        panel = load_panel(data_dir, n_tickers=1, fields=["close"])
        close = panel.get("close")
        if close is None or close.empty:
            return None
        return _median_bar_seconds(close.index)
    except Exception as e:  # noqa: BLE001 — never block research on a bar-size probe
        log.warning("could not infer bar size (%s) — prompts feed-agnostic", e)
        return None


def run_research_session(
    session_id: str | None = None,
    cutoff_date: date | None = None,
    n_papers: int = 2,
    n_ideas: int = 3,
    horizon: int = 6,
    n_tickers: int | None = 15,
    reset: bool = False,
    paper_sample_strategy: str = "unread_first",
    data_dir: str = "ticker_data",
    dedup_scope: str = "package",
    allowed_fields: list[str] | None = None,
    force_prediction_horizon: bool = False,
) -> dict[str, Any]:
    """Run one Factor Researcher session and persist any kept factors.

    ``paper_sample_strategy`` (``"unread_first"`` | ``"random"``) controls how the
    session's papers are picked.  Combined with the scenario-scoped read-log
    (``PAPER_READ_LOG``), ``"unread_first"`` advances deterministically through the
    library across sessions; ``"random"`` samples it.

    ``dedup_scope`` (``"package"`` | ``"prerun"``) controls what new factor ids are
    de-duped against during brainstorm: every registered class ∪ the active DB, or
    only the active prerun's own factors (so a prerun is not anchored by others').

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

    # Restrict the researcher to the fields the configured feed can actually
    # supply (LOBSTER order-book level + the fundamentals opt-out), so it only
    # invents factors this run can serve.  Computed once from the active config
    # unless the caller passes an explicit set; falls back to un-gated (None) if
    # the provider/config can't be resolved.
    if allowed_fields is None:
        try:
            from quant_fund_agent.data import usable_fields

            allowed_fields = sorted(usable_fields())
        except Exception as e:  # noqa: BLE001 — never block research on field resolution
            log.warning("could not resolve usable fields (%s) — research un-gated", e)
            allowed_fields = None

    state = FactorResearcherState(
        session_id=session_id or date.today().isoformat(),
        cutoff_date=cutoff_date,
        n_papers=n_papers,
        n_factor_ideas=n_ideas,
        ic_target_horizon=horizon,
        force_prediction_horizon=force_prediction_horizon,
        seconds_per_bar=_infer_seconds_per_bar(data_dir),
        n_tickers=n_tickers,
        paper_sample_strategy=paper_sample_strategy,
        data_dir=data_dir,
        dedup_scope=dedup_scope,
        allowed_fields=allowed_fields,
    )
    log.info("Factor research session '%s' (papers=%d, ideas=%d, IC@%d recorded, "
             "not gated; prediction horizon %s; %s data fields in scope)",
             state.session_id, n_papers, n_ideas, horizon,
             f"fixed at {horizon}" if force_prediction_horizon else "free",
             len(allowed_fields) if allowed_fields is not None else "all")
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
    target_horizon: int | None = None,
    run_statistician: bool = True,
    cutoff_date: date | None = None,
    position_construction: str | None = None,
) -> StrategyPipelineResult:
    """Selector → Architect → (optionally) Statistician, in one call.

    The ``approved`` flag is True only when the architect approved *and*
    the statistician accepted (or the statistician was skipped on an
    architect approval).

    ``cutoff_date`` (walk-forward backtest only): when set, the Architect and
    Statistician only see panel data strictly **before** it — so a strategy
    designed at week *t* is never contaminated by the data it will be traded on.
    ``None`` → full history (the default / production behaviour).

    ``position_construction`` selects how the composite signal becomes positions
    for every strategy this pass builds: ``"cross_sectional"`` (dollar-neutral
    rank) or ``"per_underlying"`` (directional boundary).  ``None`` → resolve from
    the data type / ``QF_POSITION_CONSTRUCTION`` (LOBSTER → per_underlying, else
    cross_sectional); ``"auto"`` is accepted too.
    """
    from quant_fund_agent.agents.architect.graph import architect_graph
    from quant_fund_agent.agents.selector.graph import selector_graph
    from quant_fund_agent.agents.statistician.graph import statistician_graph
    from quant_fund_agent.backtesting.positions import resolve_position_construction

    as_of = cutoff_date.isoformat() if cutoff_date else None
    construction = resolve_position_construction(position_construction)
    log.info("Position construction: %s", construction)

    selector_result = selector_graph.invoke({})
    log.info("Selector hypothesis: %s", selector_result.get("hypothesis", "")[:120])
    log.info("Selected factors: %s", selector_result.get("selected_factor_ids"))

    architect_result = architect_graph.invoke({
        "hypothesis": selector_result["hypothesis"],
        "selected_factor_ids": selector_result["selected_factor_ids"],
        "factor_catalog": selector_result["factor_catalog"],
        "max_iterations": max_iterations,
        "oos_split_ratio": oos_ratio,
        # Fallback only: the Architect CHOOSES target_horizon from the selected
        # factors' own prediction horizons (mode), so this is used only when no
        # factor declares one.  An explicit value here is the fallback default.
        "target_horizon": target_horizon if target_horizon is not None else 6,
        "as_of": as_of,
        "position_construction": construction,
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
        "as_of": as_of,
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
            "position_construction": spec.position_construction,
            "position_params": dict(spec.position_params or {}),
        },
        holding_period=spec.holding_period,
        metadata={
            "n_trials": len(result.architect_result.get("trial_history", [])),
            "deflated_sharpe_ratio": deflated_sr,
            "architect_reasoning": spec.reasoning,
            "statistician_reasoning": final_reasoning,
            # Which (config, prerun) scope + data config this strategy was built
            # under — so a merged/pooled book stays auditable (see merge.py).
            "provenance": _scope_provenance(),
        },
    )
    return record, is_returns


def _scope_provenance() -> dict:
    """Provenance stamp for the active scope (``QF_SCOPE`` + live data config)."""
    import os

    from quant_fund_agent.config import get_settings, provenance_stamp

    return provenance_stamp(os.getenv("QF_SCOPE", "main"), get_settings().data)


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
            position_construction=p.get("position_construction", "cross_sectional"),
            position_params=p.get("position_params", {}),
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
        position_construction=p.get("position_construction", "cross_sectional"),
        position_params=p.get("position_params", {}),
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
    annualisation_factor: float | None = None,
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
    if annualisation_factor is None:
        annualisation_factor = _infer_annualisation(strategy_db)

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
