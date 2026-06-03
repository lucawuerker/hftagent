"""Architect Agent – LangGraph subgraph with a model-selection refinement loop.

Workflow
-------
1. **design_model** – the LLM picks a model from the pre-implemented toolbox
   (regression / tree ensembles / gradient boosting, or the static-weights
   baseline), its hyper-parameters, the feature factors, and the position
   construction settings.
2. **fit_and_backtest** – the chosen model is fit on the in-sample window and
   back-tested.  This is delegated to the **quant-modeling MCP server** (the
   server owns the heavy data panel; only JSON crosses the boundary).  ML models
   are fit on an IS-train sub-window and scored on a held-out IS-valid
   sub-window, so the reported metrics are never measured on the fitting data.
   The trial is appended to ``trial_history``.
3. **evaluate_and_decide** – the LLM reviews the metrics + fit diagnostics
   (train-vs-valid gap = overfitting) across the whole trial history and returns
   ``"approve"`` or ``"revise"``.
4a. **approve** → END (the fitted-model spec goes to the Statistician).
4b. **revise_model** → back to ``fit_and_backtest`` with a new model/params.

Every trial is recorded so the Statistician can compute the Deflated Sharpe
ratio using the total number of trials as the number of independent tests — a
wider ML search is therefore correctly penalised for multiple testing.

The held-out OOS slice is never shown to the architect; it is reserved for the
Statistician, which reloads the persisted fitted model (``model_artifact_path``)
and re-runs it on OOS without refitting.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from quant_fund_agent.agents.architect.state import (
    ArchitectState,
    StrategySpec,
    TrialRecord,
)
from quant_fund_agent.modeling import (
    STATIC_WEIGHTS,
    available_model_types,
)

log = logging.getLogger("architect")

LLM_MODEL = os.getenv("ARCHITECT_LLM_MODEL", "gpt-4o-mini")
DATA_DIR = os.getenv("DATA_DIR", "ticker_data")


# Optional universe cap (shared with the modeling service via ARCHITECT_N_TICKERS).
# ``ticker_data/`` is several GB across 50+ tickers in production; loading the
# whole panel routinely OOMs a Jupyter kernel.  Set ARCHITECT_N_TICKERS=10 (or
# similar) to bound peak memory.  Unset → load every ticker found.
def _resolve_n_tickers() -> int | None:
    raw = os.getenv("ARCHITECT_N_TICKERS")
    if not raw:
        return None
    try:
        n = int(raw)
        return n if n > 0 else None
    except ValueError:
        log.warning("Ignoring invalid ARCHITECT_N_TICKERS=%r (need positive int).", raw)
        return None


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=0.4)


def _parse_json(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        return json.loads(content[start:end])


def _as_int(value: Any, default: int) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ── prompt rendering helpers ──────────────────────────────────────────

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


def _model_menu_text() -> str:
    from quant_fund_agent.mcp import client

    lines: list[str] = []
    for m in client.list_models():
        lines.append(f"- {m['model_type']} ({m['family']}): {m['description']}")
        for p in m.get("params", []):
            if "min" in p and "max" in p:
                rng = f", range [{p['min']}, {p['max']}]"
            elif "choices" in p:
                rng = f", choices {p['choices']}"
            else:
                rng = ""
            lines.append(f"    {p['name']} ({p['type']}, default={p['default']}{rng})")
    return "\n".join(lines)


def _trial_history_text(state: ArchitectState) -> str:
    if not state.trial_history:
        return "(no prior trials)"
    lines = []
    for t in state.trial_history:
        m = t.metrics
        diag = m.get("diagnostics") or {}
        feats = t.spec.factor_ids or list(t.spec.weights.keys())
        lines.append(
            f"Trial {t.iteration}: model={t.spec.model_type} "
            f"Sharpe={m.get('sharpe_ratio')} MaxDD={m.get('max_drawdown')} "
            f"AnnRet={m.get('annualised_return')} Sortino={m.get('sortino_ratio')} "
            f"HitRate={m.get('hit_rate')} IC={m.get('ic_mean')} "
            f"trainR2={diag.get('train_r2')} validR2={diag.get('valid_r2')}\n"
            f"  params={t.spec.model_params} target_h={t.spec.target_horizon} "
            f"features={feats}\n"
            f"  hold={t.spec.holding_period} maxpos={t.spec.max_positions} "
            f"eqw={t.spec.equal_weight} minconv={t.spec.min_conviction}"
            + (f"\n  ERROR: {m['error']}" if m.get("error") else "")
            + f"\n  reasoning: {t.spec.reasoning[:150]}"
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
    out = [f"  {k}: {metrics.get(k)}" for k in keys]
    diag = metrics.get("diagnostics") or {}
    if diag:
        out.append(f"  [fit] train_r2={diag.get('train_r2')} "
                   f"valid_r2={diag.get('valid_r2')} "
                   f"n_train={diag.get('n_train_samples')} "
                   f"n_valid={diag.get('n_valid_samples')}")
        if diag.get("feature_importances"):
            out.append(f"  [fit] feature_importances={diag['feature_importances']}")
    if metrics.get("error"):
        out.append(f"  ERROR: {metrics['error']}")
    return "\n".join(out)


# ── spec parsing (shared by design + revise) ──────────────────────────

def _parse_spec(parsed: dict, state: ArchitectState, fallback_name: str) -> StrategySpec:
    """Turn an LLM JSON response into a validated ``StrategySpec``.

    Model hyper-parameters are NOT validated here — the modeling service clamps
    them server-side against the catalog ranges.  We only validate the things
    the architect owns: the model_type and the factor universe.
    """
    valid_ids = set(state.selected_factor_ids)

    model_type = (parsed.get("model_type") or STATIC_WEIGHTS).strip()
    if model_type not in available_model_types():
        log.warning("Unknown/unavailable model_type %r from LLM — falling back to %s.",
                    model_type, STATIC_WEIGHTS)
        model_type = STATIC_WEIGHTS

    weights = {k: _as_float(v, 0.0) for k, v in (parsed.get("weights") or {}).items()
               if k in valid_ids}
    factor_ids = [f for f in (parsed.get("factor_ids") or []) if f in valid_ids]

    if model_type == STATIC_WEIGHTS:
        if not weights:  # default to an equal-weight blend of every selected factor
            weights = {fid: 1.0 for fid in valid_ids}
        factor_ids = list(weights.keys())
    else:
        if not factor_ids:  # default to using every selected factor as a feature
            factor_ids = list(valid_ids)

    model_params = parsed.get("model_params")
    if not isinstance(model_params, dict):
        model_params = {}

    return StrategySpec(
        strategy_name=parsed.get("strategy_name", fallback_name),
        model_type=model_type,
        model_params=model_params,
        factor_ids=factor_ids,
        target_horizon=state.target_horizon,  # config-driven, not LLM-chosen
        weights=weights,
        holding_period=_as_int(parsed.get("holding_period", state.target_horizon),
                               state.target_horizon),
        max_positions=_as_int(parsed.get("max_positions", 20), 20),
        equal_weight=bool(parsed.get("equal_weight", False)),
        min_conviction=_as_float(parsed.get("min_conviction", 0.0), 0.0),
        reasoning=parsed.get("reasoning", ""),
    )


# ── node 1: design model (first iteration) ───────────────────────────

DESIGN_PROMPT = """\
You are a senior quantitative portfolio manager building a systematic strategy.

TRADING HYPOTHESIS
------------------
{hypothesis}

AVAILABLE FACTORS (use these as model features / inputs)
-------------------------------------------------------
{factor_details}

MODEL TOOLBOX (choose exactly one model_type)
---------------------------------------------
{model_menu}

FORECAST HORIZON
----------------
Fixed at {target_horizon} bars: the model predicts {target_horizon}-bar-ahead
returns and positions are held ~{target_horizon} bars.  Choose the model and
features for THIS horizon.

Your task:
Design a strategy that predicts forward returns from the factors using ONE model
from the toolbox, then trades that prediction.  Prefer the simplest model that
fits the hypothesis; reach for tree/boosting models only when you expect
non-linear interactions between factors.

Decide:
1. strategy_name: short descriptive name (2-5 words).
2. model_type: one of the toolbox model_type values.
3. model_params: object of hyper-parameters for the chosen model (use the listed
   parameter names; out-of-range values are clamped). Use {{}} to accept defaults.
4. factor_ids: which available factors to use as features (a subset is fine).
   Omit for static_weights.
5. weights: ONLY for static_weights — {{"factor_id": weight, ...}} (a negative
   weight reverses that factor).
6. holding_period: bars to hold before re-scoring.  Default to the forecast
   horizon ({target_horizon} bars) unless you have a clear reason to differ.
8. max_positions: how many stocks to trade at once (default 20, max 50).
9. equal_weight: true to give every selected position equal size.
10. min_conviction: minimum absolute z-score of the combined signal to take a
    position (0.0 = no filter, 0.5 = moderate).

Respond in JSON:
  "strategy_name": string,
  "model_type": string,
  "model_params": object,
  "factor_ids": [string, ...],
  "weights": object,
  "holding_period": int,
  "max_positions": int,
  "equal_weight": bool,
  "min_conviction": float,
  "reasoning": "short paragraph explaining the model + feature choices"
"""


def design_model(state: ArchitectState) -> dict:
    """First iteration: the LLM picks a model and configures the strategy."""
    llm = _get_llm()
    resp = llm.invoke(DESIGN_PROMPT.format(
        hypothesis=state.hypothesis,
        factor_details=_factor_details_text(state),
        model_menu=_model_menu_text(),
        target_horizon=state.target_horizon,
    ))
    spec = _parse_spec(_parse_json(resp.content), state, fallback_name="unnamed")
    log.info("[design] model_type=%s features=%s", spec.model_type,
             spec.factor_ids or list(spec.weights.keys()))
    return {
        "strategy_spec": spec,
        "iteration": 1,
        "initial_reasoning": spec.reasoning,
    }


# ── node 2: fit + backtest (delegated to the modeling MCP server) ─────

def fit_and_backtest(state: ArchitectState) -> dict:
    """Fit the chosen model on IS and backtest it via the modeling MCP server.

    On any fit/backtest failure the trial is recorded with an ``error`` (and no
    metrics) so the loop can revise instead of crashing the whole pipeline.
    """
    from quant_fund_agent.mcp import client

    spec = state.strategy_spec
    feats = spec.factor_ids or list(spec.weights.keys())
    log.info("[iter %d] fit+backtest model=%s features=%s …",
             state.iteration, spec.model_type, feats)

    t0 = time.time()
    try:
        result = client.fit_and_backtest(
            model_type=spec.model_type,
            factor_ids=spec.factor_ids,
            model_params=spec.model_params,
            weights=spec.weights,
            target_horizon=spec.target_horizon,
            holding_period=spec.holding_period,
            max_positions=spec.max_positions,
            equal_weight=spec.equal_weight,
            min_conviction=spec.min_conviction,
            oos_split_ratio=state.oos_split_ratio,
            strategy_id=f"arch_{uuid.uuid4().hex[:8]}",
            as_of=state.as_of,
        )
        metrics = result.get("metrics", {})
        metrics["diagnostics"] = result.get("diagnostics", {})
        spec.model_artifact_path = result.get("artifact_path", "") or ""
        log.info("[iter %d] done in %.1fs — Sharpe=%s (train_r2=%s valid_r2=%s)",
                 state.iteration, time.time() - t0, metrics.get("sharpe_ratio"),
                 (result.get("diagnostics") or {}).get("train_r2"),
                 (result.get("diagnostics") or {}).get("valid_r2"))
    except Exception as e:  # pragma: no cover — keep the loop alive
        log.warning("[iter %d] fit+backtest failed: %s", state.iteration, e)
        metrics = {"error": str(e), "sharpe_ratio": None}
        spec.model_artifact_path = ""

    trial = TrialRecord(iteration=state.iteration, spec=spec, metrics=metrics)
    new_history = list(state.trial_history) + [trial]
    return {
        "strategy_spec": spec,
        "backtest_metrics": metrics,
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
model_type: {model_type}
{metrics_summary}

FULL TRIAL HISTORY
------------------
{trial_history}

Review the results.  Consider:
- Sharpe ratio: above 1.0 is decent, above 2.0 is strong (note: this is measured
  on a held-out in-sample validation slice).
- Max drawdown: smaller magnitude is better.
- Hit rate: above 0.50 suggests edge.
- Overfitting: a large gap between train_r2 and valid_r2 (train high, valid low
  or negative) means the model memorised noise — prefer a simpler model, more
  regularisation, or fewer features.
- Whether earlier iterations were better (are changes improving things?).
- You have {remaining} iterations left.  If this is the last iteration you MUST
  approve (the best trial will be kept automatically).

Decide:
- "approve" if results are acceptable or you've exhausted useful changes.
- "revise" if a specific change (different model_type, hyper-parameters, feature
  set, or position settings) would plausibly improve results.

Respond in JSON:
  "decision": "approve" or "revise",
  "reasoning": "paragraph explaining your decision"
"""


def _trial_score(t: TrialRecord) -> float | None:
    """Leakage-free selection score for a trial.

    Prefer the held-out **validation rank-IC** (computed inside the IS window on
    data the model wasn't fitted on); fall back to in-sample Sharpe for the
    static-weights baseline, which has no fitted model / validation score.
    """
    if t.metrics.get("error"):
        return None
    vs = t.metrics.get("validation_score")
    if vs is None:
        vs = (t.metrics.get("diagnostics") or {}).get("valid_ic")
    if vs is None:
        vs = t.metrics.get("sharpe_ratio")
    try:
        return float(vs) if vs is not None else None
    except (TypeError, ValueError):
        return None


def _best_trial(state: ArchitectState) -> TrialRecord | None:
    """The successful trial with the best selection score (for forced approval)."""
    scored = [(s, t) for t in state.trial_history if (s := _trial_score(t)) is not None]
    if not scored:
        return None
    return max(scored, key=lambda x: x[0])[1]


def evaluate_and_decide(state: ArchitectState) -> dict:
    """LLM reviews metrics + diagnostics and decides to approve or revise."""
    remaining = state.max_iterations - state.iteration

    if remaining <= 0:
        best = _best_trial(state)
        if best is not None:
            log.info("[iter %d] Max iterations reached — approving best trial "
                     "(iter %d, Sharpe=%s).", state.iteration, best.iteration,
                     best.metrics.get("sharpe_ratio"))
            return {
                "decision": "approve",
                "decision_reasoning": "Maximum iterations reached. Kept the best trial.",
                "strategy_spec": best.spec,
                "backtest_metrics": best.metrics,
            }
        log.info("[iter %d] Max iterations reached — approving (no successful trial).",
                 state.iteration)
        return {
            "decision": "approve",
            "decision_reasoning": "Maximum iterations reached.",
        }

    llm = _get_llm()
    resp = llm.invoke(EVALUATE_PROMPT.format(
        hypothesis=state.hypothesis,
        iteration=state.iteration,
        max_iterations=state.max_iterations,
        model_type=state.strategy_spec.model_type,
        metrics_summary=_metrics_summary_text(state.backtest_metrics),
        trial_history=_trial_history_text(state),
        remaining=remaining,
    ))
    parsed = _parse_json(resp.content)

    decision = parsed.get("decision", "approve").lower().strip()
    if decision not in ("approve", "revise"):
        decision = "approve"

    # Never approve a current trial that errored if a revise is still possible.
    if decision == "approve" and state.backtest_metrics.get("error"):
        decision = "revise"

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

MODEL TOOLBOX
-------------
{model_menu}

FULL TRIAL HISTORY (all previous attempts)
-------------------------------------------
{trial_history}

YOUR PREVIOUS REVIEW
--------------------
{decision_reasoning}

Revise the strategy based on what the trials show.  You may change the
model_type, model_params, factor_ids (features), or the position settings (the
forecast horizon is fixed).  Use ONLY the factors listed above.  Make a targeted change that
addresses the weakness you identified (e.g. if train_r2 >> valid_r2, regularise
more or use fewer features / a simpler model).  Avoid repeating a configuration
you've already tried.

Respond in JSON (same schema as the design step):
  "strategy_name": string,
  "model_type": string,
  "model_params": object,
  "factor_ids": [string, ...],
  "weights": object,
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
        model_menu=_model_menu_text(),
        trial_history=_trial_history_text(state),
        decision_reasoning=state.decision_reasoning,
    ))
    spec = _parse_spec(_parse_json(resp.content), state,
                       fallback_name=state.strategy_spec.strategy_name)

    new_iteration = state.iteration + 1
    log.info("[iter %d → %d] Revised: model_type=%s features=%s",
             state.iteration, new_iteration, spec.model_type,
             spec.factor_ids or list(spec.weights.keys()))
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
    graph.add_node("fit_and_backtest", fit_and_backtest)
    graph.add_node("evaluate_and_decide", evaluate_and_decide)
    graph.add_node("revise_model", revise_model)

    graph.add_edge(START, "design_model")
    graph.add_edge("design_model", "fit_and_backtest")
    graph.add_edge("fit_and_backtest", "evaluate_and_decide")
    graph.add_conditional_edges(
        "evaluate_and_decide",
        _route_after_evaluate,
        {
            "approved": END,
            "revise": "revise_model",
        },
    )
    graph.add_edge("revise_model", "fit_and_backtest")

    return graph


architect_graph = build_architect_graph().compile()


# ---------------------------------------------------------------------------
# Cached panel / signal helpers — kept here because the Statistician imports
# them to build its OOS test context.  (The architect's own fit path now runs
# inside the modeling MCP server; these are only used in-process by the
# statistician stage.)
# ---------------------------------------------------------------------------

_PANEL_CACHE: dict | None = None
_SIGNAL_CACHE: dict[str, object] = {}


def _load_panel_cached() -> dict:
    global _PANEL_CACHE
    if _PANEL_CACHE is None:
        from quant_fund_agent.backtesting.data_loader import load_panel
        n_tickers = _resolve_n_tickers()
        log.info("Loading data from %s%s …", DATA_DIR,
                 f" (capped at {n_tickers} tickers)" if n_tickers else "")
        t0 = time.time()
        _PANEL_CACHE = load_panel(DATA_DIR, n_tickers=n_tickers)
        log.info("Data loaded in %.1fs (universe size: %d)", time.time() - t0,
                 next(iter(_PANEL_CACHE.values())).shape[1] if _PANEL_CACHE else 0)
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
