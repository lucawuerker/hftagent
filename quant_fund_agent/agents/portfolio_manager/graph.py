"""Portfolio Manager Agent — LangGraph subgraph.

Pipeline
--------
1. ``load_universe``        – pull every approved strategy, its KPIs,
                              correlation matrix, covariance matrix.
2. ``monitor_performance``  – for every LIVE strategy compare live to
                              in-sample metrics; flag drift, retire if
                              broken.  In ACTIVE mode the LLM decides;
                              in SELECTOR mode rule-based thresholds
                              from the personality are used.
3. ``screen_strategies``    – apply personality hard filters (min
                              Sharpe, max DD, max correlation) and pick
                              the strategy roster.
4. ``construct_portfolio``  – run the chosen construction method.
                              SELECTOR mode uses
                              ``profile.default_construction_method``
                              (or an override).  ACTIVE mode lets the
                              LLM pick the method (or even the weights
                              directly via ``custom_llm``).
5. ``finalise``             – build PortfolioAllocation list + expected
                              metrics, persist a ``PortfolioRecord`` to
                              the portfolio DB, update each
                              ``StrategyRecord.pm_status`` (LIVE /
                              PAUSED / FLAGGED / RETIRED).

The graph is identical regardless of mode — the *behaviour* of the
``monitor`` and ``construct`` nodes branches on ``state.mode``.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

import pandas as pd
from langgraph.graph import END, START, StateGraph

from quant_fund_agent.agents.portfolio_manager.prompts import (
    CONSTRUCTION_PROMPT,
    MONITOR_PROMPT,
    SCREEN_PROMPT,
)
from quant_fund_agent.agents.portfolio_manager.state import PortfolioManagerState
from quant_fund_agent.mcp import portfolio_client
from quant_fund_agent.portfolio import get_personality_profile
from quant_fund_agent.portfolio.personalities import PersonalityProfile
from quant_fund_agent.schemas import (
    ConstructionMethod,
    PMMode,
    PMStatus,
    PortfolioAllocation,
    PortfolioRecord,
    StrategyRecord,
)

log = logging.getLogger("portfolio_manager")

LLM_MODEL = os.getenv("PM_LLM_MODEL", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _get_llm():
    """Lazy LLM construction so importing the module doesn't require an API key."""
    from langchain_openai import ChatOpenAI
    # Finite timeout + retries so a stalled response-body read can't hang the
    # pipeline forever inside SSL_read (langchain-openai's default timeout=None
    # = wait forever).  See the selector graph for the full explanation.
    return ChatOpenAI(model=LLM_MODEL, temperature=0.2, timeout=120, max_retries=4)


def _parse_json(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        return json.loads(content[start:end])


def _resolve_profile(state: PortfolioManagerState) -> PersonalityProfile:
    if state.profile is not None:
        return state.profile
    return get_personality_profile(state.personality)


def _target_n(state: PortfolioManagerState, profile: PersonalityProfile) -> int:
    return state.target_n_strategies or profile.target_n_strategies


def _strategy_to_row(rec: StrategyRecord) -> dict[str, Any]:
    """Compact dict for prompts and screening logic."""
    is_m = rec.backtest_metrics
    oos_m = rec.oos_backtest_metrics
    live_m = rec.live_metrics
    return {
        "strategy_id": rec.id,
        "name": rec.name,
        "pm_status": rec.pm_status.value,
        "is_sharpe": is_m.sharpe_ratio if is_m else None,
        "is_max_dd": is_m.max_drawdown if is_m else None,
        "is_ann_vol": is_m.annualised_volatility if is_m else None,
        "is_ann_ret": is_m.annualised_return if is_m else None,
        "is_hit_rate": is_m.hit_rate if is_m else None,
        "oos_sharpe": oos_m.sharpe_ratio if oos_m else None,
        "oos_max_dd": oos_m.max_drawdown if oos_m else None,
        "live_sharpe": live_m.sharpe_ratio if live_m else None,
        "live_max_dd": live_m.max_drawdown if live_m else None,
        "n_factors": len(rec.factor_ids),
    }


# ---------------------------------------------------------------------------
# Node 1 — load universe
# ---------------------------------------------------------------------------

def load_universe(state: PortfolioManagerState) -> dict:
    """Pull every non-retired strategy and read correlations/cov.

    The correlation / covariance matrices are lazy-cached on the
    ``StrategyDatabase`` and only recompute when something downstream
    actually changed (new strategy added, returns extended, etc.).  In
    the steady-state of a long backtest these are O(1) reads.
    """
    profile = _resolve_profile(state)
    db = state.strategy_db

    available = [
        rec for rec in db.list_strategies()
        if rec.pm_status != PMStatus.RETIRED
    ]

    universe_summary = [_strategy_to_row(rec) for rec in available]
    cov = db.covariance_matrix(annualisation_factor=state.annualisation_factor)
    corr = db.correlation_matrix

    log.info(
        "[%s] Universe: %d strategies (excluding retired); "
        "correlation cache %s.",
        state.pm_name, len(available),
        "hit" if not db.is_correlation_cache_dirty() else "miss → recomputed",
    )

    return {
        "profile": profile,
        "universe_summary": universe_summary,
        "correlation_matrix": corr,
        "covariance_matrix": cov,
    }


# ---------------------------------------------------------------------------
# Node 2 — monitor live performance
# ---------------------------------------------------------------------------

def _rule_based_monitor(
    deployed: list[StrategyRecord],
    profile: PersonalityProfile,
) -> list[dict[str, Any]]:
    """SELECTOR-mode monitor: pure-threshold logic from the personality."""
    actions: list[dict[str, Any]] = []
    for rec in deployed:
        live = rec.live_metrics
        if live is None:
            actions.append({
                "strategy_id": rec.id,
                "action": "keep",
                "reasoning": "no live metrics yet — keep observing",
            })
            continue

        live_sharpe = live.sharpe_ratio if live.sharpe_ratio is not None else 0.0
        live_dd = live.max_drawdown if live.max_drawdown is not None else 0.0

        if live_sharpe < profile.live_sharpe_kill_floor or live_dd < profile.live_dd_floor * 1.5:
            actions.append({
                "strategy_id": rec.id, "action": "retire",
                "reasoning": (
                    f"live Sharpe {live_sharpe:.2f} < kill floor "
                    f"{profile.live_sharpe_kill_floor:.2f} "
                    f"or live DD {live_dd:.2%} far below tolerance"
                ),
            })
        elif live_sharpe < profile.live_sharpe_floor or live_dd < profile.live_dd_floor:
            actions.append({
                "strategy_id": rec.id, "action": "flag",
                "reasoning": (
                    f"live Sharpe {live_sharpe:.2f} or DD {live_dd:.2%} "
                    f"breached personality floors — review"
                ),
            })
        else:
            actions.append({
                "strategy_id": rec.id, "action": "keep",
                "reasoning": "live metrics within tolerance",
            })
    return actions


def _llm_monitor(
    deployed: list[StrategyRecord],
    profile: PersonalityProfile,
    pm_name: str,
) -> list[dict[str, Any]]:
    """ACTIVE-mode monitor: let the LLM decide using personality context."""
    if not deployed:
        return []
    rows = [_strategy_to_row(rec) for rec in deployed]
    table = "\n".join(
        f"- {r['strategy_id']} | {r['name']} | "
        f"IS Sharpe={r['is_sharpe']} DD={r['is_max_dd']} | "
        f"Live Sharpe={r['live_sharpe']} DD={r['live_max_dd']}"
        for r in rows
    )
    llm = _get_llm()
    resp = llm.invoke(MONITOR_PROMPT.format(
        pm_name=pm_name,
        personality=profile.personality.value,
        personality_description=profile.description,
        live_sharpe_floor=profile.live_sharpe_floor,
        live_sharpe_kill_floor=profile.live_sharpe_kill_floor,
        live_dd_floor=profile.live_dd_floor,
        deployed_table=table,
    ))
    parsed = _parse_json(resp.content)
    valid_ids = {rec.id for rec in deployed}
    return [a for a in parsed.get("actions", []) if a.get("strategy_id") in valid_ids]


def monitor_performance(state: PortfolioManagerState) -> dict:
    """Walk the LIVE strategies and decide keep / flag / retire."""
    profile = state.profile or _resolve_profile(state)
    db = state.strategy_db

    deployed = [
        rec for rec in db.list_strategies(pm_status=PMStatus.LIVE)
    ]
    log.info("[%s] Monitoring %d live strategies (mode=%s).",
             state.pm_name, len(deployed), state.mode.value)

    if not deployed:
        return {"monitor_actions": [], "flagged_strategy_ids": [], "retired_strategy_ids": []}

    if state.mode == PMMode.ACTIVE:
        try:
            actions = _llm_monitor(deployed, profile, state.pm_name)
        except Exception as e:  # pragma: no cover — never abort the agent
            log.exception("LLM monitor failed (%s); falling back to rule-based.", e)
            actions = _rule_based_monitor(deployed, profile)
    else:
        actions = _rule_based_monitor(deployed, profile)

    flagged: list[str] = []
    retired: list[str] = []
    for a in actions:
        sid = a["strategy_id"]
        if a["action"] == "flag":
            flagged.append(sid)
            if not state.propose_only:
                db.flag_strategy(sid, reason=a.get("reasoning", "flagged by PM"))
        elif a["action"] == "retire":
            retired.append(sid)
            if not state.propose_only:
                db.retire_strategy(sid, reason=a.get("reasoning", "retired by PM"))

    return {
        "monitor_actions": actions,
        "flagged_strategy_ids": flagged,
        "retired_strategy_ids": retired,
    }


# ---------------------------------------------------------------------------
# Node 3 — screen strategies (eligible candidates for the new portfolio)
# ---------------------------------------------------------------------------

def _screen_profile_dict(profile: PersonalityProfile) -> dict[str, Any]:
    """Personality params the quant-portfolio screen tool needs."""
    return {
        "min_sharpe": profile.min_sharpe,
        "max_drawdown": profile.max_drawdown,
        "min_hit_rate": profile.min_hit_rate,
        "max_pairwise_correlation": profile.max_pairwise_correlation,
    }


def _llm_screen(
    rows: list[dict[str, Any]],
    profile: PersonalityProfile,
    target_n: int,
    corr: pd.DataFrame,
) -> tuple[list[str], str]:
    universe_table = "\n".join(
        f"- {r['strategy_id']} | {r['name']} | "
        f"Sharpe={r['is_sharpe']} DD={r['is_max_dd']} "
        f"Vol={r['is_ann_vol']} Hit={r['is_hit_rate']}"
        for r in rows
    )
    if corr.empty:
        corr_table = "(no pairwise correlations available)"
    else:
        head = corr.iloc[:10, :10].round(2)
        corr_table = head.to_string()

    llm = _get_llm()
    resp = llm.invoke(SCREEN_PROMPT.format(
        personality_description=profile.description,
        min_sharpe=profile.min_sharpe,
        max_drawdown=profile.max_drawdown,
        max_correlation=profile.max_pairwise_correlation,
        target_n=target_n,
        universe_table=universe_table,
        correlation_table=corr_table,
    ))
    parsed = _parse_json(resp.content)
    valid = {r["strategy_id"] for r in rows}
    selected = [s for s in parsed.get("selected_strategy_ids", []) if s in valid]
    return selected, parsed.get("rationale", "")


def screen_strategies(state: PortfolioManagerState) -> dict:
    """Pick the roster of strategies for the new portfolio.

    The personality hard-filter + greedy correlation-aware pick run on the
    quant-portfolio MCP server; ACTIVE mode lets the LLM override, falling back
    to the server's greedy pick when the LLM produces nothing usable.
    """
    profile = state.profile or _resolve_profile(state)
    target_n = _target_n(state, profile)

    rows = state.universe_summary
    # Don't consider already-retired or freshly-flagged strategies.
    rows = [
        r for r in rows
        if r["pm_status"] not in (PMStatus.RETIRED.value, PMStatus.FLAGGED_FOR_REVIEW.value)
    ]

    screen = portfolio_client.screen_strategies(
        rows, _screen_profile_dict(profile), target_n, state.correlation_matrix,
    )
    eligible_ids = set(screen["eligible_strategy_ids"])

    if not eligible_ids:
        log.warning("[%s] No strategies passed personality filters.", state.pm_name)
        return {"selected_strategy_ids": [], "screen_rationale": "no eligible strategies"}

    greedy_selected = screen["selected_strategy_ids"]

    if state.mode == PMMode.ACTIVE:
        eligible_rows = [r for r in rows if r["strategy_id"] in eligible_ids]
        try:
            selected, rationale = _llm_screen(
                eligible_rows, profile, target_n, state.correlation_matrix,
            )
            if not selected:
                # LLM returned nothing usable — fall back to greedy logic.
                raise RuntimeError("LLM screen returned no valid IDs")
        except Exception as e:
            log.warning("[%s] LLM screen failed (%s); falling back to greedy.",
                        state.pm_name, e)
            selected = greedy_selected
            rationale = (
                f"Greedy correlation-aware pick (Sharpe-ranked, capped at "
                f"|ρ| <= {profile.max_pairwise_correlation})."
            )
    else:
        selected = greedy_selected
        rationale = (
            f"SELECTOR mode: top-{target_n} by IS Sharpe with diversification cap "
            f"|ρ| <= {profile.max_pairwise_correlation} and personality filters "
            f"(min_sharpe={profile.min_sharpe}, max_dd={profile.max_drawdown})."
        )

    log.info("[%s] Selected %d strategies: %s",
             state.pm_name, len(selected), selected)
    return {"selected_strategy_ids": selected, "screen_rationale": rationale}


# ---------------------------------------------------------------------------
# Node 4 — construct the portfolio
# ---------------------------------------------------------------------------

def _expected_returns_from_records(
    strategy_ids: list[str],
    db,
) -> dict[str, float]:
    """Use IS annualised returns as the simplest expected-return estimate."""
    mu: dict[str, float] = {}
    for sid in strategy_ids:
        rec = db.get_strategy(sid)
        if rec is None or rec.backtest_metrics is None:
            mu[sid] = 0.0
            continue
        mu[sid] = float(rec.backtest_metrics.annualised_return or 0.0)
    return mu


def _llm_construction(
    state: PortfolioManagerState,
    profile: PersonalityProfile,
) -> tuple[ConstructionMethod, dict[str, float], str]:
    """Ask the LLM to pick a construction method (and weights if custom)."""
    rows = [r for r in state.universe_summary if r["strategy_id"] in state.selected_strategy_ids]
    selected_table = "\n".join(
        f"- {r['strategy_id']} | {r['name']} | "
        f"Sharpe={r['is_sharpe']} Vol={r['is_ann_vol']} DD={r['is_max_dd']}"
        for r in rows
    )

    llm = _get_llm()
    resp = llm.invoke(CONSTRUCTION_PROMPT.format(
        selected_table=selected_table,
        risk_aversion=profile.risk_aversion,
        default_method=profile.default_construction_method.value,
        max_weight=profile.max_weight_per_strategy,
        allow_short=profile.allow_short,
    ))
    parsed = _parse_json(resp.content)

    method_raw = parsed.get("construction_method", profile.default_construction_method.value)
    try:
        method = ConstructionMethod(method_raw)
    except ValueError:
        log.warning("Unknown method %r from LLM — falling back to default.", method_raw)
        method = profile.default_construction_method

    raw_weights: dict[str, float] = {}
    if method == ConstructionMethod.CUSTOM_LLM:
        valid = set(state.selected_strategy_ids)
        weights = parsed.get("weights", {})
        raw_weights = {
            k: float(v) for k, v in weights.items()
            if k in valid and isinstance(v, (int, float))
        }
        # L1-normalise.
        total = sum(abs(v) for v in raw_weights.values())
        if total > 0:
            raw_weights = {k: v / total for k, v in raw_weights.items()}

    return method, raw_weights, parsed.get("rationale", "")


def construct_portfolio_node(state: PortfolioManagerState) -> dict:
    """Run the chosen construction method on the selected strategies."""
    profile = state.profile or _resolve_profile(state)

    if not state.selected_strategy_ids:
        return {"chosen_method": ConstructionMethod.EQUAL_WEIGHT,
                "raw_weights": {},
                "pm_rationale": "no strategies passed screening — empty portfolio"}

    if state.mode == PMMode.ACTIVE:
        try:
            method, llm_weights, rationale = _llm_construction(state, profile)
        except Exception as e:
            log.warning("[%s] LLM construction failed (%s); using profile default.",
                        state.pm_name, e)
            method = profile.default_construction_method
            llm_weights = {}
            rationale = f"ACTIVE LLM failed ({e!s}); fell back to {method.value}."
    else:
        method = state.construction_method_override or profile.default_construction_method
        llm_weights = {}
        rationale = (
            f"SELECTOR mode: using personality default method '{method.value}' "
            f"with target_n={_target_n(state, profile)} strategies."
        )

    if method == ConstructionMethod.CUSTOM_LLM and llm_weights:
        weights = llm_weights
    else:
        expected = _expected_returns_from_records(state.selected_strategy_ids, state.strategy_db)
        result = portfolio_client.construct_portfolio(
            method=method.value,
            strategy_ids=state.selected_strategy_ids,
            cov=state.covariance_matrix,
            expected_returns=expected,
            risk_aversion=profile.risk_aversion,
            risk_free_rate=profile.risk_free_rate,
            allow_short=profile.allow_short,
            max_weight=profile.max_weight_per_strategy,
        )
        weights = result["weights"]
        method = ConstructionMethod(result["method"])

    log.info("[%s] Constructed portfolio with method=%s (%d strategies).",
             state.pm_name, method.value, len(weights))
    return {
        "chosen_method": method,
        "raw_weights": weights,
        "pm_rationale": rationale,
    }


# ---------------------------------------------------------------------------
# Node 5 — finalise: persist PortfolioRecord, update StrategyRecord.pm_status
# ---------------------------------------------------------------------------

def finalise(state: PortfolioManagerState) -> dict:
    """Build the PortfolioRecord, persist it, and update strategy PM statuses."""
    profile = state.profile or _resolve_profile(state)
    db = state.strategy_db
    pdb = state.portfolio_db

    allocations = [
        PortfolioAllocation(
            strategy_id=sid,
            weight=float(w),
            enabled=abs(float(w)) > 1e-6,
            rationale=f"{state.chosen_method.value} weight",
        )
        for sid, w in state.raw_weights.items()
    ]
    # Sort by absolute weight, descending, for readability.
    allocations.sort(key=lambda a: -abs(a.weight))

    expected = _expected_returns_from_records(
        [a.strategy_id for a in allocations], db,
    )
    metrics = portfolio_client.expected_portfolio_metrics(
        {a.strategy_id: a.weight for a in allocations},
        expected_returns=expected,
        cov=state.covariance_matrix,
    ) if not state.covariance_matrix.empty else {}

    # Update each strategy's pm_status:
    # - allocated → LIVE
    # - previously LIVE but now de-selected → PAUSED
    # Skipped when propose_only=True so the committee can do this once
    # against the consensus instead of each PM stepping on each other.
    new_live_ids = {a.strategy_id for a in allocations if a.enabled}
    if not state.propose_only:
        for rec in db.list_strategies():
            if rec.pm_status == PMStatus.RETIRED:
                continue
            if rec.pm_status == PMStatus.FLAGGED_FOR_REVIEW:
                continue  # leave flagged ones to the research team
            if rec.id in new_live_ids:
                db.set_pm_status(rec.id, PMStatus.LIVE)
            elif rec.pm_status == PMStatus.LIVE:
                db.set_pm_status(
                    rec.id, PMStatus.PAUSED,
                    notes=f"De-selected by {state.pm_name} on this rebalance.",
                )

    record = PortfolioRecord(
        id=f"portfolio_{uuid.uuid4().hex[:10]}",
        pm_name=state.pm_name,
        mode=state.mode,
        personality=profile.personality,
        construction_method=state.chosen_method,
        target_n_strategies=_target_n(state, profile),
        allocations=allocations,
        flagged_strategy_ids=list(state.flagged_strategy_ids),
        retired_strategy_ids=list(state.retired_strategy_ids),
        expected_metrics=metrics,
        pm_rationale=(
            (state.screen_rationale + "\n\n" if state.screen_rationale else "")
            + state.pm_rationale
        ).strip(),
        metadata={
            "monitor_actions": state.monitor_actions,
            "annualisation_factor": state.annualisation_factor,
            "universe_size": len(state.universe_summary),
            "propose_only": state.propose_only,
        },
    )
    if not state.propose_only:
        pdb.add_portfolio(record)

    log.info(
        "[%s] Portfolio %s finalised: %d allocations, method=%s, expected_vol=%s",
        state.pm_name, record.id, len(allocations),
        record.construction_method.value, metrics.get("expected_volatility"),
    )

    return {
        "allocations": allocations,
        "expected_metrics": metrics,
        "portfolio_record": record,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_portfolio_manager_graph() -> StateGraph:
    graph = StateGraph(PortfolioManagerState)

    graph.add_node("load_universe", load_universe)
    graph.add_node("monitor_performance", monitor_performance)
    graph.add_node("screen_strategies", screen_strategies)
    graph.add_node("construct_portfolio", construct_portfolio_node)
    graph.add_node("finalise", finalise)

    graph.add_edge(START, "load_universe")
    graph.add_edge("load_universe", "monitor_performance")
    graph.add_edge("monitor_performance", "screen_strategies")
    graph.add_edge("screen_strategies", "construct_portfolio")
    graph.add_edge("construct_portfolio", "finalise")
    graph.add_edge("finalise", END)

    return graph


portfolio_manager_graph = build_portfolio_manager_graph().compile()
