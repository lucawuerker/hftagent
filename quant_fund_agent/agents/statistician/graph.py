"""Statistician Agent — runs statistical tests on an approved strategy.

Pipeline
--------
1. **build_context** – reconstruct the full test context from the
   architect's state (deserialise the IS portfolio returns, load the
   full panel for the OOS test, pre-compute factor signals on the full
   panel, carry through the trial history for deflated SR).
2. **run_mandatory_tests** – execute every test registered with
   ``is_mandatory=True`` (currently: deflated Sharpe ratio + OOS test).
3. **assess_risk_and_pick_optional** – the LLM reads the strategy spec
   and the mandatory-test results and decides which additional
   registered-but-optional tests to run (MVP: no optional tests exist
   yet, so this node is a no-op but the scaffolding is in place for the
   future statistician-researcher agents).
4. **run_optional_tests** – run each picked optional test.
5. **final_decision** – the LLM synthesises all test results into
   ``accept`` or ``reject`` with a written justification.

Simplifications (MVP)
---------------------
- No optional tests exist yet; the node runs but has nothing to do.
- The final decision is LLM-based with a rule-based safety net
  (if any mandatory test has verdict=FAIL we force ``reject``).
- The statistician does NOT yet write to the StrategyDatabase.
"""

from __future__ import annotations

import json
import logging
import os

import pandas as pd
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from quant_fund_agent.agents.statistician.state import StatisticianState
from quant_fund_agent.statistics import (
    StatTestContext,
    StatTestResult,
    StatTestVerdict,
    discover_tests,
    get_all_test_classes,
    get_mandatory_test_ids,
    get_optional_test_ids,
    instantiate_test,
)

log = logging.getLogger("statistician")

LLM_MODEL = os.getenv("STATISTICIAN_LLM_MODEL", "gpt-4o-mini")
DATA_DIR = os.getenv("DATA_DIR", "ticker_data")


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=0.2)


def _parse_json(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        return json.loads(content[start:end])


def _deserialise_series(payload: dict | None) -> pd.Series | None:
    if not payload or "timestamps" not in payload:
        return None
    idx = pd.to_datetime(payload["timestamps"])
    return pd.Series(payload["values"], index=idx)


def _test_result_to_text(r: StatTestResult) -> str:
    return (
        f"- [{r.verdict.value.upper()}] {r.test_name} "
        f"(id={r.test_id}, value={r.value}, threshold={r.threshold})\n"
        f"    {r.summary}"
    )


# ── node 1: build the test context (heavy work: load data, compute signals) ──

# Reuse the architect's cached panel / signal helpers so we don't load twice.
from quant_fund_agent.agents.architect.graph import (  # noqa: E402
    _factor_signal_cached,
    _load_panel_cached,
)


def _build_context(state: StatisticianState) -> StatTestContext:
    from quant_fund_agent.factors import discover_factors

    discover_factors()

    data_full = _load_panel_cached()
    factor_signals_full = {
        fid: _factor_signal_cached(fid, data_full)
        for fid in state.strategy_spec.weights
    }

    is_portfolio_returns = _deserialise_series(
        state.backtest_metrics.get("portfolio_returns")
    )
    is_equity_curve = _deserialise_series(
        state.backtest_metrics.get("equity_curve")
    )

    return StatTestContext(
        strategy_spec=state.strategy_spec,
        hypothesis=state.hypothesis,
        initial_reasoning=state.initial_reasoning,
        final_reasoning=state.final_reasoning,
        is_portfolio_returns=is_portfolio_returns,
        is_equity_curve=is_equity_curve,
        is_metrics=state.backtest_metrics,
        data_full=data_full,
        factor_signals_full=factor_signals_full,
        oos_split_ratio=state.oos_split_ratio,
        trial_history=state.trial_history,
        bars_per_day=state.bars_per_day,
    )


# ── node 2: run mandatory tests ──────────────────────────────────────

def run_mandatory_tests(state: StatisticianState) -> dict:
    discover_tests()

    mandatory = state.mandatory_test_ids or get_mandatory_test_ids()
    log.info("Running %d mandatory test(s): %s", len(mandatory), mandatory)

    ctx = _build_context(state)
    results: list[StatTestResult] = []
    for test_id in mandatory:
        test = instantiate_test(test_id)
        log.info("  … %s", test.name)
        try:
            results.append(test.run(ctx))
        except Exception as e:  # pragma: no cover — don't crash the agent
            log.exception("Test %s raised: %s", test_id, e)
            results.append(StatTestResult(
                test_id=test_id,
                test_name=test.name,
                verdict=StatTestVerdict.FAIL,
                summary=f"Test raised an exception: {e}",
            ))

    return {"test_results": results}


# ── node 3: LLM picks optional tests based on perceived risk ─────────

ASSESS_PROMPT = """\
You are the quantitative fund's statistician.  You have received a
candidate strategy and its mandatory-test results.  Decide how "risky"
this strategy is and pick any optional registered tests you think are
worth running.

TRADING HYPOTHESIS
------------------
{hypothesis}

STRATEGY SPEC
-------------
{spec_json}

IN-SAMPLE METRICS
-----------------
{metrics_summary}

MANDATORY TEST RESULTS
----------------------
{mandatory_results}

AVAILABLE OPTIONAL TESTS
------------------------
{optional_catalog}

Consider how many trials the architect ran (overfitting risk), how
extreme the in-sample metrics look, and whether any mandatory test
returned WARN or FAIL.  If there are no optional tests yet, return an
empty list and say so in the reasoning.

Respond in JSON:
  "risk_level": "low" | "medium" | "high",
  "selected_test_ids": ["test_id", ...],
  "reasoning": "short paragraph"
"""


def _spec_json(state: StatisticianState) -> str:
    s = state.strategy_spec
    return json.dumps({
        "strategy_name": s.strategy_name,
        "weights": s.weights,
        "holding_period": s.holding_period,
        "max_positions": s.max_positions,
        "equal_weight": s.equal_weight,
        "min_conviction": s.min_conviction,
        "n_trials": len(state.trial_history),
    }, indent=2)


def _metrics_summary(state: StatisticianState) -> str:
    keys = [
        "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "annualised_return", "annualised_volatility", "max_drawdown",
        "hit_rate", "profit_factor", "avg_daily_turnover",
        "ic_mean", "ic_ir",
    ]
    m = state.backtest_metrics
    return "\n".join(f"  {k}: {m.get(k)}" for k in keys)


def _optional_catalog() -> str:
    opts = get_optional_test_ids()
    if not opts:
        return "(none registered yet)"
    classes = get_all_test_classes()
    return "\n".join(f"- {tid}: {classes[tid].description}" for tid in opts)


def _mandatory_results_text(results: list[StatTestResult]) -> str:
    if not results:
        return "(none)"
    return "\n".join(_test_result_to_text(r) for r in results)


def assess_risk_and_pick_optional(state: StatisticianState) -> dict:
    optional = get_optional_test_ids()
    if not optional and not state.forced_optional_test_ids:
        log.info("No optional tests registered — skipping risk assessment.")
        return {
            "risk_assessment": "No optional tests registered yet.",
            "selected_optional_test_ids": [],
        }

    llm = _get_llm()
    resp = llm.invoke(ASSESS_PROMPT.format(
        hypothesis=state.hypothesis,
        spec_json=_spec_json(state),
        metrics_summary=_metrics_summary(state),
        mandatory_results=_mandatory_results_text(state.test_results),
        optional_catalog=_optional_catalog(),
    ))
    parsed = _parse_json(resp.content)

    selected = [t for t in parsed.get("selected_test_ids", []) if t in optional]
    # caller can force extra tests regardless of the LLM's choice
    selected = list(dict.fromkeys(selected + state.forced_optional_test_ids))

    log.info("Risk: %s. Optional tests selected: %s",
             parsed.get("risk_level", "?"), selected)

    return {
        "risk_assessment": parsed.get("reasoning", ""),
        "selected_optional_test_ids": selected,
    }


# ── node 4: run the selected optional tests ──────────────────────────

def run_optional_tests(state: StatisticianState) -> dict:
    if not state.selected_optional_test_ids:
        return {}

    ctx = _build_context(state)
    new_results: list[StatTestResult] = list(state.test_results)
    for test_id in state.selected_optional_test_ids:
        test = instantiate_test(test_id)
        log.info("Optional test: %s", test.name)
        try:
            new_results.append(test.run(ctx))
        except Exception as e:  # pragma: no cover
            log.exception("Test %s raised: %s", test_id, e)
            new_results.append(StatTestResult(
                test_id=test_id,
                test_name=test.name,
                verdict=StatTestVerdict.FAIL,
                summary=f"Test raised an exception: {e}",
            ))

    return {"test_results": new_results}


# ── node 5: final decision ───────────────────────────────────────────

DECISION_PROMPT = """\
You are the quantitative fund's statistician.  Based on ALL the
statistical-test results below, decide whether to ACCEPT or REJECT the
candidate strategy.

TRADING HYPOTHESIS
------------------
{hypothesis}

STRATEGY SPEC
-------------
{spec_json}

ARCHITECT'S ORIGINAL RATIONALE
------------------------------
{initial_reasoning}

ARCHITECT'S FINAL RATIONALE
---------------------------
{final_reasoning}

IN-SAMPLE METRICS
-----------------
{metrics_summary}

ALL TEST RESULTS
----------------
{all_results}

Rules of thumb:
- Any test with verdict=FAIL should very strongly push toward REJECT.
- Multiple WARN verdicts are also cause for caution.
- A deflated Sharpe ratio below 0.80 is a strong rejection signal.
- If the OOS Sharpe is negative, reject.

Respond in JSON:
  "decision": "accept" | "reject",
  "reasoning": "short paragraph"
"""


def final_decision(state: StatisticianState) -> dict:
    # Rule-based safety net: any mandatory FAIL → reject.
    any_fail = any(
        r.verdict == StatTestVerdict.FAIL for r in state.test_results
    )

    llm = _get_llm()
    resp = llm.invoke(DECISION_PROMPT.format(
        hypothesis=state.hypothesis,
        spec_json=_spec_json(state),
        initial_reasoning=state.initial_reasoning or "(not provided)",
        final_reasoning=state.final_reasoning or "(not provided)",
        metrics_summary=_metrics_summary(state),
        all_results="\n".join(_test_result_to_text(r) for r in state.test_results),
    ))
    parsed = _parse_json(resp.content)

    decision = parsed.get("decision", "reject").lower().strip()
    if decision not in ("accept", "reject"):
        decision = "reject"
    if any_fail and decision == "accept":
        log.warning("Overriding LLM 'accept' to 'reject' due to a FAILed test.")
        decision = "reject"

    log.info("Statistician final decision: %s", decision.upper())
    return {
        "final_decision": decision,
        "final_reasoning_statistician": parsed.get("reasoning", ""),
    }


# ── graph construction ────────────────────────────────────────────────

def build_statistician_graph() -> StateGraph:
    graph = StateGraph(StatisticianState)

    graph.add_node("run_mandatory_tests", run_mandatory_tests)
    graph.add_node("assess_risk_and_pick_optional", assess_risk_and_pick_optional)
    graph.add_node("run_optional_tests", run_optional_tests)
    graph.add_node("final_decision", final_decision)

    graph.add_edge(START, "run_mandatory_tests")
    graph.add_edge("run_mandatory_tests", "assess_risk_and_pick_optional")
    graph.add_edge("assess_risk_and_pick_optional", "run_optional_tests")
    graph.add_edge("run_optional_tests", "final_decision")
    graph.add_edge("final_decision", END)

    return graph


statistician_graph = build_statistician_graph().compile()
