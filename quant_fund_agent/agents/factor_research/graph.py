"""Factor Researcher Agent – LangGraph subgraph.

This agent expands the seed factor ensemble during a simulation.  In a
single *research session* it:

  1. ``load_papers``       – pick N papers (respecting ``cutoff_date``
                              to avoid lookahead bias) and extract text
                              from their PDFs.
  2. ``brainstorm``        – one LLM call *per paper* (read that single
                              paper, propose ideas from it), so no call
                              ever has to fit every paper in its context
                              and each idea is grounded in one paper.
                              The session-wide ``n_factor_ideas`` budget
                              is split across papers and the pooled ideas
                              are deduped + trimmed to that budget.
  3. ``generate_code``     – one LLM call per idea: produce a complete
                              ``BaseFactor`` subclass.  Bad code is
                              dropped without aborting the session.
  4. ``backtest_factors``  – run the standard single-factor IC backtest
                              for every materialised candidate.  This
                              validates that the factor actually runs on
                              the panel and records its IC for reference;
                              it is **not** an accept/reject gate.
  5. ``filter_and_persist`` – keep every candidate that materialised and
                              backtested successfully (IC magnitude is
                              *not* a gate — a low-IC factor such as
                              volatility can still be a useful feature in
                              combination); tag survivors with
                              ``source=RESEARCHER`` and the session id;
                              drop only the ones whose code failed to
                              generate or errored in the backtest (both
                              from the registry and the filesystem).

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
import math
import os
import time

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from quant_fund_agent.agents.factor_research.codegen import purge_researcher_code
from quant_fund_agent.agents.factor_research.prompts import (
    BRAINSTORM_PROMPT,
    CODEGEN_PROMPT,
    EXAMPLE_FACTOR,
    MAX_PREDICTION_HORIZON,
    OPERATOR_REFERENCE,
    RETRY_FEEDBACK,
    build_data_context,
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

# gpt-4o-mini exposes a 128k-token context window.  A single dense paper
# (math / tables / LaTeX often tokenises at ~1 token per character) can on
# its own blow past that, so brainstorm caps the *input* tokens it will send
# in one call and splits the paper body across several calls when needed,
# leaving the remaining headroom for the JSON response + tokeniser slack.
BRAINSTORM_MAX_INPUT_TOKENS = int(
    os.getenv("BRAINSTORM_MAX_INPUT_TOKENS", "110000")
)


def _get_llm(temperature: float = 0.6):
    # Built through the provider-agnostic factory so a prerun can pick the
    # research model/provider via FACTOR_RESEARCH_LLM_MODEL / _PROVIDER (read at
    # call time).  Finite timeout + retries so a stalled response-body read can't
    # hang the pipeline forever (langchain's default timeout=None = wait forever).
    from quant_fund_agent.llm import make_chat_llm

    return make_chat_llm(temperature=temperature, timeout=120, max_retries=4)


def _coerce_horizon(raw, default: int) -> int:
    """Best-effort coerce an LLM-supplied horizon to a valid bar count.

    Falls back to ``default`` on anything non-numeric / non-positive and clamps
    to the guardrail range.  The codegen validator re-checks the *class
    attribute* (the source of truth); this only sanitises the prompt spec.
    """
    try:
        h = int(raw)
    except (TypeError, ValueError):
        return default
    if h <= 0:
        return default
    return min(h, MAX_PREDICTION_HORIZON)


def _coerce_horizon_list(raw) -> list[int]:
    """Coerce an LLM-supplied ``suggested_horizons`` list, dropping bad entries."""
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[int] = []
    for el in raw:
        try:
            h = int(el)
        except (TypeError, ValueError):
            continue
        if 0 < h <= MAX_PREDICTION_HORIZON:
            out.append(h)
    return out


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

def _existing_factor_ids(scope: str = "package") -> list[str]:
    """Factor ids a new brainstorm should avoid, per ``scope``.

    ``"package"`` (default) → all registered factor classes ∪ the active factor
    DB; ``"prerun"`` → only the active prerun's factor DB (so a prerun is not
    anchored by other preruns' factors).  Computed by the quant-research MCP
    server so the registry it queries is the same process that materialises new
    factors.
    """
    return research_client.existing_factor_ids(scope=scope)


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


_TOKEN_ENCODER: object | None = None


def _encoder():
    """Lazily resolve a tiktoken encoder for ``LLM_MODEL``.

    Cached across calls.  Returns ``None`` (and we fall back to a
    conservative char-based estimate) if tiktoken is unavailable or has
    no mapping for the model.
    """
    global _TOKEN_ENCODER
    if _TOKEN_ENCODER is None:
        try:
            import tiktoken

            try:
                _TOKEN_ENCODER = tiktoken.encoding_for_model(LLM_MODEL)
            except KeyError:
                _TOKEN_ENCODER = tiktoken.get_encoding("o200k_base")
        except Exception:  # tiktoken missing / failed to load
            _TOKEN_ENCODER = False  # sentinel: don't retry the import
    return _TOKEN_ENCODER or None


def _count_tokens(text: str) -> int:
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text))
    # Without tiktoken, assume the worst case seen on dense PDFs (~1 token
    # per char) so we under-fill rather than overflow the context window.
    return len(text)


def _split_text_by_tokens(text: str, max_tokens: int) -> list[str]:
    """Split ``text`` into contiguous pieces each <= ``max_tokens`` tokens.

    Slicing on token boundaries (then decoding back) guarantees every
    piece fits, which a char-based split cannot promise for dense text.
    """
    max_tokens = max(1, max_tokens)
    enc = _encoder()
    if enc is None:
        # Conservative char fallback: ~1 token/char, so chunk by chars.
        return [text[i:i + max_tokens] for i in range(0, len(text), max_tokens)] or [""]
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return [text]
    return [enc.decode(tokens[i:i + max_tokens])
            for i in range(0, len(tokens), max_tokens)]


def _fixed_horizon(state: FactorResearcherState) -> int | None:
    """Run-level horizon if this session forces one, otherwise ``None``."""
    return int(state.ic_target_horizon) if state.force_prediction_horizon else None


def _render_brainstorm_prompt(
    paper: PaperSnippet | None,
    n_ideas: int,
    known_ids: set[str],
    data_context: str,
) -> str:
    block = _papers_block([paper] if paper else [])
    return BRAINSTORM_PROMPT.format(
        n_ideas=n_ideas,
        data_context=data_context,
        papers_block=block,
        existing_ids=", ".join(sorted(known_ids)) if known_ids else "(none)",
    )


def _invoke_brainstorm(
    llm: ChatOpenAI,
    prompt: str,
    tag: str,
    n_ideas: int,
    session_id: str,
) -> list[dict]:
    t0 = time.time()
    resp = llm.invoke(prompt)
    log.info("[session %s] brainstorm call (%s, asked %d, %d tok): %.1fs",
             session_id, tag, n_ideas, _count_tokens(prompt), time.time() - t0)
    return _parse_json(resp.content).get("ideas", [])


def _brainstorm_one(
    llm: ChatOpenAI,
    paper: PaperSnippet | None,
    n_ideas: int,
    known_ids: set[str],
    session_id: str,
    data_context: str,
) -> list[dict]:
    """Brainstorm ideas over a *single* paper (or own knowledge).

    ``paper=None`` runs the knowledge-only fallback (used when a session
    has no papers).  ``known_ids`` (existing factors + ideas already
    accepted this session) is passed so the model avoids id collisions.

    The whole paper text is read.  If a paper is so long that the prompt
    would exceed the model's input budget, its body is split across the
    fewest calls that each fit (two in the common case) and the ideas are
    pooled — so a giant PDF never trips ``context_length_exceeded``.
    Returns the raw idea dicts; validation/dedupe is the caller's job so
    it can carry state across papers.
    """
    tag = paper.paper_id if paper else "own-knowledge"

    prompt = _render_brainstorm_prompt(paper, n_ideas, known_ids, data_context)
    # Common path: the whole paper fits in one call.  (Knowledge-only has
    # no paper body to split, so it always goes here.)
    if paper is None or _count_tokens(prompt) <= BRAINSTORM_MAX_INPUT_TOKENS:
        return _invoke_brainstorm(llm, prompt, tag, n_ideas, session_id)

    # Too long: budget the paper body = total input cap minus the fixed
    # prompt scaffolding (rendered once with an empty body), then split the
    # body on token boundaries so every part's prompt is guaranteed to fit.
    overhead = _count_tokens(
        _render_brainstorm_prompt(paper.model_copy(update={"text": ""}),
                                  n_ideas, known_ids, data_context)
    )
    # Small reserve absorbs the per-part "(part i/n)" title suffix and the
    # fact that token counts aren't exactly additive across a split point.
    body_budget = BRAINSTORM_MAX_INPUT_TOKENS - overhead - 256
    parts = _split_text_by_tokens(paper.text, body_budget)
    n = len(parts)
    log.info("[session %s] paper %s too long (%d tok) — splitting body into "
             "%d part(s) so no single message overflows the context window",
             session_id, tag, _count_tokens(paper.text), n)

    ideas: list[dict] = []
    for i, body in enumerate(parts, start=1):
        part_paper = paper.model_copy(
            update={"text": body, "title": f"{paper.title} (part {i}/{n})"}
        )
        part_prompt = _render_brainstorm_prompt(
            part_paper, n_ideas, known_ids, data_context)
        ideas.extend(_invoke_brainstorm(
            llm, part_prompt, f"{tag} part {i}/{n}", n_ideas, session_id))
    return ideas


def brainstorm(state: FactorResearcherState) -> dict:
    """Brainstorm one paper at a time → pooled, deduped list[FactorIdea].

    Each paper gets its own LLM call (so a call never has to fit every
    paper in its context, and ideas stay grounded in a single paper).
    The session-wide ``n_factor_ideas`` budget is divided across the
    selected papers; the pooled ideas are deduped (against existing
    factors and against each other) and trimmed to the budget.  With no
    papers we make a single knowledge-only call for the whole budget.
    """
    existing = _existing_factor_ids(scope=state.dedup_scope)
    known_ids: set[str] = set(existing)  # grows as ideas are accepted
    seen: set[str] = set()
    ideas: list[FactorIdea] = []

    # Only describe the fields this run can actually supply, so brainstormed
    # ideas stay within the configured data scope (LOBSTER level / fundamentals).
    # The bar size lets the LLM reason about prediction horizons in wall-clock time.
    data_context = build_data_context(
        state.allowed_fields, state.seconds_per_bar, _fixed_horizon(state))

    budget = state.n_factor_ideas
    papers = state.selected_papers
    if papers:
        per_paper = max(1, math.ceil(budget / len(papers)))
        plan: list[tuple[PaperSnippet | None, int]] = [(p, per_paper) for p in papers]
    else:
        plan = [(None, budget)]

    log.info("[session %s] brainstorming up to %d idea(s) across %d paper(s), "
             "one call each (dedup against %s, %d known id(s)) …",
             state.session_id, budget, len(papers), state.dedup_scope, len(existing))

    llm = _get_llm(temperature=0.7)
    for paper, n_ask in plan:
        if len(ideas) >= budget:
            break
        raw_ideas = _brainstorm_one(
            llm, paper, n_ask, known_ids, state.session_id, data_context)
        for raw in raw_ideas:
            if len(ideas) >= budget:
                break
            fid = (raw.get("factor_id") or "").strip().lower()
            if not fid or fid in seen or fid in known_ids:
                log.warning("dropping idea with bad/collision id: %r", fid)
                continue
            seen.add(fid)
            known_ids.add(fid)  # so later papers don't reuse it
            forced_h = _fixed_horizon(state)
            ideas.append(FactorIdea(
                factor_id=fid,
                name=raw.get("name", fid),
                category=raw.get("category", "other"),
                trading_idea=raw.get("trading_idea", ""),
                description=raw.get("description", ""),
                prediction_horizon=(
                    forced_h if forced_h is not None else
                    _coerce_horizon(raw.get("prediction_horizon"),
                                    state.ic_target_horizon)
                ),
                suggested_horizons=(
                    [forced_h] if forced_h is not None else
                    _coerce_horizon_list(raw.get("suggested_horizons"))
                ),
                # The idea came from this paper's call, so attribute it
                # there authoritatively (knowledge-only: trust the model).
                source_paper_ids=([paper.paper_id] if paper
                                  else raw.get("source_paper_ids", [])),
            ))

    log.info("[session %s] kept %d idea(s) from %d paper-call(s)",
             state.session_id, len(ideas), len(plan))
    return {"factor_ideas": ideas}


# ---------------------------------------------------------------------------
# Node 3: generate_code (per idea, in a plain Python loop)
# ---------------------------------------------------------------------------

def _codegen_prompt(idea: FactorIdea, data_context: str, feedback: str = "") -> str:
    """Render the codegen prompt for one idea, optionally with retry feedback.

    On the first try ``feedback`` is empty so the FEEDBACK block in
    the template collapses to nothing.  On a retry we pass the error
    message from the failed attempt so the LLM can self-correct.
    """
    feedback_block = RETRY_FEEDBACK.format(error=feedback) if feedback else ""
    return CODEGEN_PROMPT.format(
        data_context=data_context,
        operator_reference=OPERATOR_REFERENCE,
        example_factor=EXAMPLE_FACTOR,
        factor_id=idea.factor_id,
        name=idea.name,
        category=idea.category,
        trading_idea=idea.trading_idea,
        description=idea.description,
        prediction_horizon=idea.prediction_horizon,
        suggested_horizons=idea.suggested_horizons or [idea.prediction_horizon],
        feedback_block=feedback_block,
    )


def _try_codegen_once(
    llm: ChatOpenAI,
    idea: FactorIdea,
    data_context: str,
    feedback: str = "",
    expected_prediction_horizon: int | None = None,
) -> tuple[str | None, str]:
    """One LLM round-trip + materialise.

    Returns ``(code_path, "")`` on success or ``(None, error_message)``
    on failure — the error message is suitable for feeding straight
    back into a retry prompt.
    """
    try:
        resp = llm.invoke(_codegen_prompt(idea, data_context, feedback=feedback))
        parsed = _parse_json(resp.content)
        code = parsed.get("code", "")
    except Exception as e:
        return None, f"codegen LLM call failed: {e}"

    idea.code = code  # keep for audit even if materialise fails
    result = research_client.materialise_factor(
        idea.factor_id, code,
        expected_prediction_horizon=expected_prediction_horizon)
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

    # Same field-scoped context as brainstorm, so generated code only reads
    # fields this run can supply (out-of-scope factors are also gated at persist).
    data_context = build_data_context(
        state.allowed_fields, state.seconds_per_bar, _fixed_horizon(state))
    expected_horizon = _fixed_horizon(state)

    for idea in state.factor_ideas:
        code_path, error = _try_codegen_once(
            llm, idea, data_context,
            expected_prediction_horizon=expected_horizon)
        attempts = 1
        while code_path is None and attempts <= max_retries:
            log.info("[%s] codegen attempt %d failed (%s) — retrying with feedback",
                     idea.factor_id, attempts, error)
            code_path, error = _try_codegen_once(
                llm, idea, data_context, feedback=error,
                expected_prediction_horizon=expected_horizon)
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
    """Persist every successfully-backtested candidate; tag them.

    IC magnitude is **not** a gate: a factor with a low standalone IC (e.g.
    volatility) can still be a valuable feature in combination, so the
    decision of whether to *use* a factor is left to the downstream agents
    (Selector / Architect), not gated here.  A candidate is kept as long as
    it materialised into runnable code and the IC backtest ran without
    erroring; its IC at the target horizon is recorded for reference only.

    Survivors are added to the factor DB tagged as
    ``FactorSource.RESEARCHER`` with the current ``research_session_id``.
    Rejects (codegen / backtest *failures* only) are removed from the
    registry AND from the filesystem so nothing broken lingers to mislead
    the next session.  The actual DB write and filesystem purge happen on
    the quant-research MCP server.
    """
    kept_records: list[dict] = []
    rejected: list[dict] = []

    # Tag every factor with the research model that produced it, so named preruns
    # are comparable and a factor's origin model is recoverable for analysis.
    from quant_fund_agent.llm import resolve_research_model, resolve_research_provider

    llm_model = resolve_research_model()
    llm_provider = resolve_research_provider(llm_model)

    for cand in state.candidates:
        fid = cand.idea.factor_id
        ic = _ic_at_horizon(cand.backtest_metrics, state.ic_target_horizon)
        passes = (
            cand.code_path                       # was materialised
            and cand.backtest_metrics is not None  # IC backtest ran without erroring
        )

        # Resolve the factor's declared inputs once (for the scope gate below
        # and for the persisted provenance / required_tier).
        from quant_fund_agent.factors.registry import get_factor_class

        _cls = get_factor_class(fid) if cand.code_path else None
        _inputs = list(getattr(_cls, "inputs", ["close"]) or ["close"]) if _cls else ["close"]
        # Prediction horizon: the validated class attribute is canonical; fall
        # back to the idea's choice, then the session's reference horizon.
        _horizon = int(getattr(_cls, "prediction_horizon", None)
                       or cand.idea.prediction_horizon or state.ic_target_horizon)
        _sug = list(getattr(_cls, "suggested_horizons", None)
                    or cand.idea.suggested_horizons or [])

        # Out-of-scope gate: drop a factor that reads a field this run's data
        # feed cannot supply (e.g. a Level-3 microstructure field on a Level-2
        # LOBSTER config, or a fundamental field with fundamentals turned off),
        # even if its IC backtest happened to run.  ``allowed_fields is None``
        # disables the gate (un-scoped legacy callers).
        if passes and state.allowed_fields is not None:
            from quant_fund_agent.data.tiers import SYNTHESIZED_FIELDS, is_compatible

            if not is_compatible(_inputs, set(state.allowed_fields)):
                out_of_scope = sorted(
                    set(_inputs) - set(state.allowed_fields) - set(SYNTHESIZED_FIELDS)
                )
                passes = False
                cand.rejected_reason = (
                    f"uses fields outside the configured data scope: {out_of_scope}"
                )

        if passes:
            # Capture the factor's data requirements for provider gating.
            from quant_fund_agent.data.tiers import required_tier

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
                required_inputs=_inputs,
                required_tier=required_tier(_inputs),
                prediction_horizon=_horizon,
                suggested_horizons=_sug,
                metadata={
                    "ic_target_horizon": state.ic_target_horizon,
                    "ic_at_target": ic,
                    "llm_model": llm_model,
                    "llm_provider": llm_provider,
                },
            )
            kept_records.append(record.model_dump(mode="json"))
            ic_txt = f"{abs(ic):.4f}" if ic is not None else "NA"
            log.info("[%s] KEPT  (|IC@%d|=%s, IC not gated)",
                     fid, state.ic_target_horizon, ic_txt)
        else:
            rejected.append({"factor_id": fid, "code_path": cand.code_path})
            reason = cand.rejected_reason or "did not produce backtest metrics"
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
