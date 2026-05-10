"""Architect Agent – LangGraph subgraph with refinement loop.

Workflow
-------
1. **design_model** – LLM decides how to combine the selected factors.
2. **run_backtest** – compute factor signals, build strategy, backtest.
   The result is appended to ``trial_history``.
3. **evaluate_and_decide** – LLM reviews backtest metrics and the full
   trial history.  Returns ``"approve"`` or ``"revise"``.
4a. **approve** → END (strategy goes to the Statistician Agent).
4b. **revise_model** → back to ``run_backtest`` (loop).
    The LLM sees all prior trials so it can learn from past mistakes.
    Iteration counter is incremented each time.
    If ``iteration >= max_iterations`` the loop is force-stopped and
    the best trial so far is kept.

Every trial (spec + metrics) is recorded in ``trial_history`` so the
downstream Statistician can compute the deflated Sharpe ratio using the
total number of trials as the number of independent tests.

MVP simplifications still in place
-----------------------------------
- Only static weighted combination (no regression / ML fitting).
- LLM *chooses* the weights heuristically rather than fitting from data.
- No train/test split.
- Strategy is created in memory, not persisted as a file.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from quant_fund_agent.agents.architect.state import (
    ArchitectState,
    StrategySpec,
    TrialRecord,
)

log = logging.getLogger("architect")

LLM_MODEL = os.getenv("ARCHITECT_LLM_MODEL", "gpt-4o-mini")
DATA_DIR = os.getenv("DATA_DIR", "ticker_data")


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=0.4)


def _parse_json(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        return json.loads(content[start:end])


def _factor_details_text(state: ArchitectState) -> str:
    selected = {f.factor_id: f for f in state.factor_catalog
                if f.factor_id in state.selected_factor_ids}
    return "\n".join(
        f"- {f.factor_id} | {f.name} | cat={f.category}\n"
        f"  IC(10s)={f.ic_1} ICIR(10s)={f.icir_1} | "
        f"IC(1m)={f.ic_6} ICIR(1m)={f.icir_6} | "
        f"IC(10m)={f.ic_60} ICIR(10m)={f.icir_60}\n"
        f"  desc: {f.description}"
        for f in selected.values()
    )


def _trial_history_text(state: ArchitectState) -> str:
    if not state.trial_history:
        return "(no prior trials)"
    lines = []
    for t in state.trial_history:
        m = t.metrics
        lines.append(
            f"Trial {t.iteration}: Sharpe={m.get('sharpe_ratio')} "
            f"MaxDD={m.get('max_drawdown')} "
            f"AnnRet={m.get('annualised_return')} "
            f"AnnVol={m.get('annualised_volatility')} "
            f"Sortino={m.get('sortino_ratio')} "
            f"HitRate={m.get('hit_rate')} "
            f"IC={m.get('ic_mean')} ICIR={m.get('ic_ir')}\n"
            f"  weights={t.spec.weights} hold={t.spec.holding_period} "
            f"maxpos={t.spec.max_positions} eqw={t.spec.equal_weight} "
            f"minconv={t.spec.min_conviction}\n"
            f"  reasoning: {t.spec.reasoning[:150]}"
        )
    return "\n".join(lines)


def _metrics_summary_text(metrics: dict) -> str:
    keys = [
        "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "annualised_return", "annualised_volatility",
        "max_drawdown", "max_drawdown_duration_bars",
        "hit_rate", "profit_factor",
        "avg_daily_turnover", "avg_positions_held",
        "ic_mean", "ic_ir",
    ]
    return "\n".join(f"  {k}: {metrics.get(k)}" for k in keys)


# ── node 1: design model (first iteration) ───────────────────────────

DESIGN_PROMPT = """\
You are a senior quantitative portfolio manager.

TRADING HYPOTHESIS
------------------
{hypothesis}

SELECTED FACTORS
----------------
{factor_details}

Your task:
Design a trading strategy that combines these factors using a **static
weighted combination** (weighted sum of the factor signals).

Decide:
1. A short descriptive name for the strategy (2-5 words).
2. The weight for each factor.  Weights do not need to sum to 1 (the
   backtester normalises positions afterwards).  Use the IC/ICIR values
   and the hypothesis logic to guide your weighting.  A negative weight
   means you are *reversing* that factor's signal.
3. holding_period: how many 10-second bars to hold before re-scoring.
   Typical values: 1 (every bar), 6 (1 min), 60 (10 min).
4. max_positions: how many stocks to trade at once (default 20, max 50).
5. equal_weight: true if you want all selected positions to have equal
   weight regardless of signal strength, false otherwise.
6. min_conviction: minimum absolute z-score of the combined signal below
   which positions are set to zero (0.0 = no filter, 0.5 = moderate).

Respond in JSON:
  "strategy_name": string,
  "weights": {{"factor_id": weight, ...}},
  "holding_period": int,
  "max_positions": int,
  "equal_weight": bool,
  "min_conviction": float,
  "reasoning": "short paragraph explaining your choices"
"""


def design_model(state: ArchitectState) -> dict:
    """First iteration: LLM designs the strategy from scratch."""
    llm = _get_llm()
    resp = llm.invoke(DESIGN_PROMPT.format(
        hypothesis=state.hypothesis,
        factor_details=_factor_details_text(state),
    ))
    parsed = _parse_json(resp.content)

    valid_ids = set(state.selected_factor_ids)
    weights = {k: v for k, v in parsed.get("weights", {}).items() if k in valid_ids}

    spec = StrategySpec(
        strategy_name=parsed.get("strategy_name", "unnamed"),
        weights=weights,
        holding_period=int(parsed.get("holding_period", 6)),
        max_positions=int(parsed.get("max_positions", 20)),
        equal_weight=bool(parsed.get("equal_weight", False)),
        min_conviction=float(parsed.get("min_conviction", 0.0)),
        reasoning=parsed.get("reasoning", ""),
    )

    return {
        "strategy_spec": spec,
        "iteration": 1,
        "initial_reasoning": spec.reasoning,
    }


# ── node 2: run backtest ─────────────────────────────────────────────

# Module-level caches so we load data / compute factor signals once per
# process rather than per iteration.  Key includes the selected factor set
# so switching architect runs still recomputes correctly.
_PANEL_CACHE: dict | None = None
_SIGNAL_CACHE: dict[str, object] = {}


def _load_panel_cached() -> dict:
    global _PANEL_CACHE
    if _PANEL_CACHE is None:
        from quant_fund_agent.backtesting.data_loader import load_panel
        log.info("Loading data from %s …", DATA_DIR)
        t0 = time.time()
        _PANEL_CACHE = load_panel(DATA_DIR)
        log.info("Data loaded in %.1fs", time.time() - t0)
    return _PANEL_CACHE


def _factor_signal_cached(factor_id: str, data: dict) -> object:
    """Compute (and cache) a factor's signal on the *full* panel."""
    from quant_fund_agent.factors import instantiate_factor
    if factor_id not in _SIGNAL_CACHE:
        _SIGNAL_CACHE[factor_id] = instantiate_factor(factor_id).calc(data)
    return _SIGNAL_CACHE[factor_id]


def _serialise_series(s) -> dict:
    """Turn a pd.Series with DatetimeIndex into a JSON-friendly dict."""
    return {
        "timestamps": [ts.isoformat() for ts in s.index],
        "values": [float(v) for v in s.values],
    }


def run_backtest(state: ArchitectState) -> dict:
    """Compute factor signals on the full panel, slice to the IN-SAMPLE
    window, build the DynamicStrategy, run the backtest, and append the
    trial to history.  The OOS slice is reserved for the statistician."""
    from quant_fund_agent.backtesting.data_split import split_panel, split_signals
    from quant_fund_agent.backtesting.strategy_backtester import backtest_strategy
    from quant_fund_agent.factors import discover_factors
    from quant_fund_agent.strategies.dynamic import DynamicStrategy

    spec = state.strategy_spec

    data_full = _load_panel_cached()
    discover_factors()

    log.info("[iter %d] Computing signals for %s …",
             state.iteration, list(spec.weights.keys()))
    factor_signals_full: dict = {
        fid: _factor_signal_cached(fid, data_full) for fid in spec.weights
    }

    # Chronological IS / OOS split (architect only ever sees IS)
    data_is, _ = split_panel(data_full, state.oos_split_ratio)
    signals_is, _ = split_signals(factor_signals_full, state.oos_split_ratio)

    strategy = DynamicStrategy(
        strategy_id=f"architect_{uuid.uuid4().hex[:8]}",
        name=spec.strategy_name,
        description=state.hypothesis,
        factor_ids=list(spec.weights.keys()),
        weights=spec.weights,
        holding_period=spec.holding_period,
        max_positions=spec.max_positions,
        equal_weight=spec.equal_weight,
        min_conviction=spec.min_conviction,
    )

    log.info("[iter %d] Running in-sample backtest "
             "(OOS ratio held out: %.2f) …",
             state.iteration, state.oos_split_ratio)
    t1 = time.time()
    result = backtest_strategy(strategy, signals_is, data_is)
    elapsed = time.time() - t1
    log.info("[iter %d] Backtest done in %.1fs", state.iteration, elapsed)

    metrics_dict = result.metrics.model_dump()
    metrics_dict["backtest_elapsed_s"] = round(elapsed, 2)
    # Include the equity curve and per-bar returns so that downstream agents
    # (statistician, reports) have everything they need without re-running.
    metrics_dict["equity_curve"] = _serialise_series(result.cumulative_returns)
    metrics_dict["portfolio_returns"] = _serialise_series(result.portfolio_returns)

    trial = TrialRecord(
        iteration=state.iteration,
        spec=spec,
        metrics=metrics_dict,
    )
    new_history = list(state.trial_history) + [trial]

    return {
        "backtest_metrics": metrics_dict,
        "trial_history": new_history,
    }


# ── node 3: evaluate and decide ──────────────────────────────────────

EVALUATE_PROMPT = """\
You are a senior quantitative portfolio manager reviewing a backtest.

TRADING HYPOTHESIS
------------------
{hypothesis}

CURRENT TRIAL (iteration {iteration} of max {max_iterations})
-------------------------------------------------------------
{metrics_summary}

FULL TRIAL HISTORY
------------------
{trial_history}

Review the backtest results.  Consider:
- Sharpe ratio: above 1.0 is decent, above 2.0 is strong.
- Max drawdown: smaller magnitude is better; above -5% is concerning for
  a short intraday horizon.
- Hit rate: above 0.50 suggests the signal has edge.
- IC / ICIR: positive IC with |ICIR| > 0.05 is meaningful.
- Whether earlier iterations were better or worse (are changes improving?).
- You have {remaining} iterations left.  If this is the last iteration
  you MUST approve (choose the best trial).

Decide:
- "approve" if the results are acceptable or you've exhausted useful changes.
- "revise" if you believe a specific change to weights, holding period,
  max_positions, equal_weight, or min_conviction would improve results.

Respond in JSON:
  "decision": "approve" or "revise",
  "reasoning": "paragraph explaining your decision"
"""


def evaluate_and_decide(state: ArchitectState) -> dict:
    """LLM reviews backtest metrics and decides to approve or revise."""
    remaining = state.max_iterations - state.iteration

    if remaining <= 0:
        log.info("[iter %d] Max iterations reached — force approving.", state.iteration)
        return {
            "decision": "approve",
            "decision_reasoning": "Maximum iterations reached. Approving best result.",
        }

    llm = _get_llm()
    resp = llm.invoke(EVALUATE_PROMPT.format(
        hypothesis=state.hypothesis,
        iteration=state.iteration,
        max_iterations=state.max_iterations,
        metrics_summary=_metrics_summary_text(state.backtest_metrics),
        trial_history=_trial_history_text(state),
        remaining=remaining,
    ))
    parsed = _parse_json(resp.content)

    decision = parsed.get("decision", "approve").lower().strip()
    if decision not in ("approve", "revise"):
        decision = "approve"

    log.info("[iter %d] Decision: %s", state.iteration, decision)

    return {
        "decision": decision,
        "decision_reasoning": parsed.get("reasoning", ""),
    }


# ── node 4: revise model ─────────────────────────────────────────────

REVISE_PROMPT = """\
You are a senior quantitative portfolio manager.

TRADING HYPOTHESIS
------------------
{hypothesis}

AVAILABLE FACTORS
-----------------
{factor_details}

FULL TRIAL HISTORY (all previous attempts)
-------------------------------------------
{trial_history}

YOUR PREVIOUS REVIEW
--------------------
{decision_reasoning}

Based on the results of all previous trials, revise the strategy.
You may change any parameter: weights, holding_period, max_positions,
equal_weight, min_conviction.  You MUST use only the factors listed
above — do not add new factor IDs.

Think about what specifically didn't work and make a targeted change.
Avoid repeating a configuration you've already tried.

Respond in JSON:
  "strategy_name": string,
  "weights": {{"factor_id": weight, ...}},
  "holding_period": int,
  "max_positions": int,
  "equal_weight": bool,
  "min_conviction": float,
  "reasoning": "what you changed and why"
"""


def revise_model(state: ArchitectState) -> dict:
    """LLM revises the strategy spec based on prior trial history."""
    llm = _get_llm()
    resp = llm.invoke(REVISE_PROMPT.format(
        hypothesis=state.hypothesis,
        factor_details=_factor_details_text(state),
        trial_history=_trial_history_text(state),
        decision_reasoning=state.decision_reasoning,
    ))
    parsed = _parse_json(resp.content)

    valid_ids = set(state.selected_factor_ids)
    weights = {k: v for k, v in parsed.get("weights", {}).items() if k in valid_ids}

    spec = StrategySpec(
        strategy_name=parsed.get("strategy_name", state.strategy_spec.strategy_name),
        weights=weights,
        holding_period=int(parsed.get("holding_period", 6)),
        max_positions=int(parsed.get("max_positions", 20)),
        equal_weight=bool(parsed.get("equal_weight", False)),
        min_conviction=float(parsed.get("min_conviction", 0.0)),
        reasoning=parsed.get("reasoning", ""),
    )

    new_iteration = state.iteration + 1
    log.info("[iter %d → %d] Revised strategy: %s",
             state.iteration, new_iteration, spec.strategy_name)

    return {"strategy_spec": spec, "iteration": new_iteration}


# ── routing ───────────────────────────────────────────────────────────

def _route_after_evaluate(state: ArchitectState) -> str:
    if state.decision == "approve":
        return "approved"
    return "revise"


# ── graph construction ────────────────────────────────────────────────

def build_architect_graph() -> StateGraph:
    graph = StateGraph(ArchitectState)

    graph.add_node("design_model", design_model)
    graph.add_node("run_backtest", run_backtest)
    graph.add_node("evaluate_and_decide", evaluate_and_decide)
    graph.add_node("revise_model", revise_model)

    graph.add_edge(START, "design_model")
    graph.add_edge("design_model", "run_backtest")
    graph.add_edge("run_backtest", "evaluate_and_decide")
    graph.add_conditional_edges(
        "evaluate_and_decide",
        _route_after_evaluate,
        {
            "approved": END,
            "revise": "revise_model",
        },
    )
    graph.add_edge("revise_model", "run_backtest")

    return graph


architect_graph = build_architect_graph().compile()
