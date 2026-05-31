"""Factor Researcher Agent – LangGraph subgraph.

This agent expands the seed factor ensemble during a simulation.  In a
single *research session* it:

  1. ``load_papers``       – pick N papers (respecting ``cutoff_date``
                              to avoid lookahead bias) and extract text
                              from their PDFs.
  2. ``brainstorm``        – one LLM call: read the papers, propose K
                              distinct factor ideas with a written
                              trading thesis.
  3. ``generate_code``     – one LLM call per idea: produce a complete
                              ``BaseFactor`` subclass.  Bad code is
                              dropped without aborting the session.
  4. ``backtest_factors``  – run the standard single-factor IC backtest
                              for every materialised candidate.
  5. ``filter_and_persist`` – keep only ``|IC| ≥ ic_threshold`` at the
                              target horizon; tag survivors with
                              ``source=RESEARCHER`` and the session id;
                              drop the rest (both from the registry and
                              the filesystem).

Why this shape (one big LangGraph node per stage, with the per-idea
loop done in plain Python inside ``generate_code`` / ``backtest_factors``):

- It keeps the graph topology small and easy to reason about.
- LangGraph branching / loops are best reserved for *decision* flow
  (refinement loops).  Here we just want to fan out over a known list.
- A failure on one candidate (bad codegen, calc raises, etc.) must not
  abort the session — we log it and continue.  That is much simpler to
  express with a try/except inside one node than via graph routing.
"""

from __future__ import annotations

import json
import logging
import os
import time

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from quant_fund_agent.agents.factor_research.codegen import purge_researcher_code
from quant_fund_agent.agents.factor_research.prompts import (
    BRAINSTORM_PROMPT,
    CODEGEN_PROMPT,
    DATA_CONTEXT,
    EXAMPLE_FACTOR,
    OPERATOR_REFERENCE,
    RETRY_FEEDBACK,
)
from quant_fund_agent.agents.factor_research.state import (
    FactorCandidate,
    FactorIdea,
    FactorResearcherState,
    PaperSnippet,
)
from quant_fund_agent.mcp import research_client
from quant_fund_agent.schemas import (
    BacktestMetrics,
    FactorRecord,
    FactorSource,
    FactorStatus,
    TradingIdeaCategory,
)

log = logging.getLogger("factor_research")

LLM_MODEL = os.getenv("FACTOR_RESEARCH_LLM_MODEL", "gpt-4o-mini")
FACTOR_DB_PATH = "data/factors/factor_db.json"


def _get_llm(temperature: float = 0.6) -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=temperature)


def _parse_json(content: str) -> dict:
    """Tolerant JSON parser — first try direct, then strip a possible
    markdown fence, then fall back to outermost { ... } slice."""
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # Strip ```json ... ``` fences if present
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:]
        content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        return json.loads(content[start:end])


# ---------------------------------------------------------------------------
# Node 1: load_papers
# ---------------------------------------------------------------------------

def load_papers(state: FactorResearcherState) -> dict:
    """Choose papers for this session and extract their text (via MCP)."""
    rows = research_client.load_papers(
        n=state.n_papers,
        cutoff_date=state.cutoff_date.isoformat() if state.cutoff_date else None,
        strategy=state.paper_sample_strategy,
        max_chars=state.max_chars_per_paper,
    )
    log.info("[session %s] selected %d/%d papers: %s",
             state.session_id, len(rows), state.n_papers,
             [r["paper_id"] for r in rows])

    snippets = [PaperSnippet(**row) for row in rows]
    return {"selected_papers": snippets}


# ---------------------------------------------------------------------------
# Node 2: brainstorm
# ---------------------------------------------------------------------------

def _existing_factor_ids() -> list[str]:
    """Union of (1) registered factor classes and (2) factor_db.json IDs.

    Computed by the quant-research MCP server so the registry it queries is the
    same process that materialises and backtests new factors.
    """
    return research_client.existing_factor_ids()


def _papers_block(snippets: list[PaperSnippet]) -> str:
    if not snippets:
        return "(no papers loaded for this session — rely on your own knowledge)"
    parts: list[str] = []
    for s in snippets:
        parts.append(
            f"=== {s.title} (id={s.paper_id}, published={s.published_date}) ===\n"
            f"{s.text or '(no text extracted)'}"
        )
    return "\n\n".join(parts)


def brainstorm(state: FactorResearcherState) -> dict:
    """Single LLM call → list[FactorIdea] (metadata + trading idea, no code yet)."""
    existing = _existing_factor_ids()
    prompt = BRAINSTORM_PROMPT.format(
        n_ideas=state.n_factor_ideas,
        data_context=DATA_CONTEXT,
        papers_block=_papers_block(state.selected_papers),
        existing_ids=", ".join(existing) if existing else "(none)",
    )
    log.info("[session %s] brainstorming %d ideas …",
             state.session_id, state.n_factor_ideas)

    llm = _get_llm(temperature=0.7)
    t0 = time.time()
    resp = llm.invoke(prompt)
    log.info("[session %s] brainstorm LLM call: %.1fs",
             state.session_id, time.time() - t0)
    parsed = _parse_json(resp.content)

    raw_ideas = parsed.get("ideas", [])
    existing_set = set(existing)
    ideas: list[FactorIdea] = []
    seen: set[str] = set()
    for raw in raw_ideas:
        fid = (raw.get("factor_id") or "").strip().lower()
        if not fid or fid in seen or fid in existing_set:
            log.warning("dropping idea with bad/collision id: %r", fid)
            continue
        seen.add(fid)
        ideas.append(FactorIdea(
            factor_id=fid,
            name=raw.get("name", fid),
            category=raw.get("category", "other"),
            trading_idea=raw.get("trading_idea", ""),
            description=raw.get("description", ""),
            source_paper_ids=raw.get("source_paper_ids", []),
        ))

    log.info("[session %s] kept %d/%d brainstorm ideas after dedupe",
             state.session_id, len(ideas), len(raw_ideas))
    return {"factor_ideas": ideas}


# ---------------------------------------------------------------------------
# Node 3: generate_code (per idea, in a plain Python loop)
# ---------------------------------------------------------------------------

def _codegen_prompt(idea: FactorIdea, feedback: str = "") -> str:
    """Render the codegen prompt for one idea, optionally with retry feedback.

    On the first try ``feedback`` is empty so the FEEDBACK block in
    the template collapses to nothing.  On a retry we pass the error
    message from the failed attempt so the LLM can self-correct.
    """
    feedback_block = RETRY_FEEDBACK.format(error=feedback) if feedback else ""
    return CODEGEN_PROMPT.format(
        data_context=DATA_CONTEXT,
        operator_reference=OPERATOR_REFERENCE,
        example_factor=EXAMPLE_FACTOR,
        factor_id=idea.factor_id,
        name=idea.name,
        category=idea.category,
        trading_idea=idea.trading_idea,
        description=idea.description,
        feedback_block=feedback_block,
    )


def _try_codegen_once(
    llm: ChatOpenAI,
    idea: FactorIdea,
    feedback: str = "",
) -> tuple[str | None, str]:
    """One LLM round-trip + materialise.

    Returns ``(code_path, "")`` on success or ``(None, error_message)``
    on failure — the error message is suitable for feeding straight
    back into a retry prompt.
    """
    try:
        resp = llm.invoke(_codegen_prompt(idea, feedback=feedback))
        parsed = _parse_json(resp.content)
        code = parsed.get("code", "")
    except Exception as e:
        return None, f"codegen LLM call failed: {e}"

    idea.code = code  # keep for audit even if materialise fails
    result = research_client.materialise_factor(idea.factor_id, code)
    if not result.get("ok"):
        return None, result.get("error", "materialisation failed")
    return result["code_path"], ""


def generate_code(state: FactorResearcherState) -> dict:
    """For each idea, ask the LLM for full Python source and materialise it.

    Each idea gets one initial codegen attempt and, on failure, exactly
    one retry where the previous error message is fed back to the LLM.
    Empirically the failures are usually small API slips
    (``ts_sum(x, window=5)`` instead of ``ts_sum(x, 5)``) that the LLM
    fixes immediately once it sees the error.
    """
    candidates: list[FactorCandidate] = []
    llm = _get_llm(temperature=0.2)  # lower temp for code-gen
    max_retries = 1

    for idea in state.factor_ideas:
        code_path, error = _try_codegen_once(llm, idea)
        attempts = 1
        while code_path is None and attempts <= max_retries:
            log.info("[%s] codegen attempt %d failed (%s) — retrying with feedback",
                     idea.factor_id, attempts, error)
            code_path, error = _try_codegen_once(llm, idea, feedback=error)
            attempts += 1

        if code_path is None:
            log.warning("[%s] codegen gave up after %d attempt(s): %s",
                        idea.factor_id, attempts, error)
            candidates.append(FactorCandidate(
                idea=idea, rejected_reason=error,
            ))
            continue

        log.info("[%s] materialised on attempt %d → %s",
                 idea.factor_id, attempts, code_path)
        candidates.append(FactorCandidate(
            idea=idea, code_path=code_path,
        ))

    return {"candidates": candidates}


# ---------------------------------------------------------------------------
# Node 4: backtest_factors
# ---------------------------------------------------------------------------

def backtest_factors(state: FactorResearcherState) -> dict:
    """Run the standard IC backtest on every materialised candidate (via MCP).

    The quant-research server owns the (cached) panel, selects the required
    fields from each factor's declared ``inputs``, slices to the cutoff date and
    runs the IC backtest; here we just map its JSON results back onto the
    candidates.
    """
    materialised = [c.idea.factor_id for c in state.candidates if c.code_path]
    if not materialised:
        log.info("[session %s] no materialised candidates to backtest",
                 state.session_id)
        return {"candidates": state.candidates}

    results = research_client.backtest_factors(
        factor_ids=materialised,
        horizon=state.ic_target_horizon,
        cutoff_date=state.cutoff_date.isoformat() if state.cutoff_date else None,
        data_dir=state.data_dir,
        n_tickers=state.n_tickers,
    )

    updated: list[FactorCandidate] = []
    for cand in state.candidates:
        if not cand.code_path:
            updated.append(cand)
            continue
        res = results.get(cand.idea.factor_id)
        if res and res.get("ok"):
            cand.backtest_metrics = BacktestMetrics.model_validate(res["metrics"])
        else:
            reason = (res or {}).get("error", "backtest failed")
            log.warning("[%s] %s", cand.idea.factor_id, reason)
            cand.rejected_reason = reason
        updated.append(cand)
    return {"candidates": updated}


# ---------------------------------------------------------------------------
# Node 5: filter_and_persist
# ---------------------------------------------------------------------------

def _ic_at_horizon(metrics: BacktestMetrics | None, horizon: int) -> float | None:
    if metrics is None:
        return None
    by_h = metrics.ic_by_horizon or {}
    cell = by_h.get(str(horizon)) or {}
    return cell.get("ic")


def _coerce_category(raw: str) -> TradingIdeaCategory:
    try:
        return TradingIdeaCategory(raw.lower())
    except Exception:
        return TradingIdeaCategory.OTHER


def filter_and_persist(state: FactorResearcherState) -> dict:
    """Keep candidates whose |IC| meets the threshold; persist + tag them.

    Survivors are added to the factor DB tagged as
    ``FactorSource.RESEARCHER`` with the current ``research_session_id``.
    Rejects are removed from the registry AND from the filesystem so
    nothing lingers to mislead the next session.  The actual DB write and
    filesystem purge happen on the quant-research MCP server.
    """
    kept_records: list[dict] = []
    rejected: list[dict] = []

    for cand in state.candidates:
        fid = cand.idea.factor_id
        ic = _ic_at_horizon(cand.backtest_metrics, state.ic_target_horizon)
        passes = (
            cand.code_path                       # was materialised
            and cand.backtest_metrics is not None
            and ic is not None
            and abs(ic) >= state.ic_threshold
        )
        if passes:
            record = FactorRecord(
                id=fid,
                name=cand.idea.name,
                class_name="",  # filled by lookup if needed downstream
                description=cand.idea.description,
                trading_idea=cand.idea.trading_idea,
                category=_coerce_category(cand.idea.category),
                status=FactorStatus.BACKTESTED,
                backtest_metrics=cand.backtest_metrics,
                source_paper_ids=cand.idea.source_paper_ids,
                source=FactorSource.RESEARCHER,
                research_session_id=state.session_id,
                code_path=cand.code_path,
                metadata={
                    "ic_threshold": state.ic_threshold,
                    "ic_target_horizon": state.ic_target_horizon,
                    "ic_at_target": ic,
                },
            )
            kept_records.append(record.model_dump(mode="json"))
            log.info("[%s] KEPT  (|IC@%d|=%.4f ≥ %.4f)",
                     fid, state.ic_target_horizon, abs(ic), state.ic_threshold)
        else:
            rejected.append({"factor_id": fid, "code_path": cand.code_path})
            reason = (
                f"|IC@{state.ic_target_horizon}|="
                f"{abs(ic) if ic is not None else 'NA'} < {state.ic_threshold}"
                if cand.backtest_metrics is not None
                else (cand.rejected_reason or "did not produce backtest metrics")
            )
            log.info("[%s] REJECTED (%s)", fid, reason)

    result = research_client.persist_results(kept_records, rejected)
    log.info("[session %s] research session complete: %d kept, %d rejected",
             state.session_id, len(result["kept_factor_ids"]),
             len(result["rejected_factor_ids"]))

    return {
        "kept_factor_ids": result["kept_factor_ids"],
        "rejected_factor_ids": result["rejected_factor_ids"],
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_after_brainstorm(state: FactorResearcherState) -> str:
    """If brainstorm produced zero usable ideas, skip straight to END."""
    if not state.factor_ideas:
        log.warning("[session %s] no factor ideas produced — skipping codegen.",
                    state.session_id)
        return "skip"
    return "continue"


def _route_after_codegen(state: FactorResearcherState) -> str:
    if not any(c.code_path for c in state.candidates):
        log.warning("[session %s] no candidates survived codegen — skipping backtest.",
                    state.session_id)
        return "skip"
    return "continue"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_factor_research_graph() -> StateGraph:
    graph = StateGraph(FactorResearcherState)

    graph.add_node("load_papers", load_papers)
    graph.add_node("brainstorm", brainstorm)
    graph.add_node("generate_code", generate_code)
    graph.add_node("backtest_factors", backtest_factors)
    graph.add_node("filter_and_persist", filter_and_persist)

    graph.add_edge(START, "load_papers")
    graph.add_edge("load_papers", "brainstorm")
    graph.add_conditional_edges(
        "brainstorm",
        _route_after_brainstorm,
        {"continue": "generate_code", "skip": END},
    )
    graph.add_conditional_edges(
        "generate_code",
        _route_after_codegen,
        {"continue": "backtest_factors", "skip": "filter_and_persist"},
    )
    graph.add_edge("backtest_factors", "filter_and_persist")
    graph.add_edge("filter_and_persist", END)

    return graph


factor_research_graph = build_factor_research_graph().compile()


# ---------------------------------------------------------------------------
# Convenience helpers used by run_factor_research.py and by the
# (future) simulation orchestrator.
# ---------------------------------------------------------------------------

def reset_researcher_state() -> None:
    """Purge ALL researcher artefacts: code files + DB entries.

    Call this at the start of a fresh simulation so the 1-year backtest
    starts from the deterministic seed ensemble only.
    """
    from quant_fund_agent.databases import FactorDatabase
    from quant_fund_agent.factors.registry import _FACTOR_REGISTRY

    n_files = purge_researcher_code()
    factor_db = FactorDatabase()
    factor_db.load_from_json(FACTOR_DB_PATH)
    removed = factor_db.purge_researcher_factors()
    factor_db.save_to_json(FACTOR_DB_PATH)
    # Also drop them from the in-memory registry.
    for fid in removed:
        _FACTOR_REGISTRY.pop(fid, None)
    log.info("Reset researcher state: %d code files, %d DB records",
             n_files, len(removed))
