"""The evolutionary research loop: seed → (select → mutate → evaluate → insert)*.

One *generation* = today's linear pipeline, generalised: the LLM proposes
children of Pareto-selected parents (mutation / crossover), a cheap programmatic
jitter proposes more, and every child is scored by the deterministic
:mod:`quant_fund_agent.research_eval` harness through the quant-research MCP
seam (:func:`quant_fund_agent.mcp.research_client.evaluate_fitness` — the server
owns the cached panel; ``QF_USE_MCP=0`` runs identically in-process).

Structure (all deterministic given the seed, except the LLM calls themselves):

* **Seeding** — generation 0 comes from the *existing* brainstorm/codegen path
  (papers via the read-log-scoped MCP loader, or knowledge-only), so the
  ``oneshot`` baseline and the ``evolution`` arm start from the same generator.
* **Operators** — ``llm_semantic`` (parent + reflection brief → child),
  ``crossover`` (two parents → synthesis), ``jitter`` (windows ±10%, no LLM).
  Chosen per child by seeded RNG with configurable probabilities.
* **Evaluation** — candidates are compiled in-memory (never files/registry);
  the plateau probe's jitter variants ride along in the same call.  ``N_trials``
  is billed per *scored* candidate (a child that fails to compile never looked
  at the data, so it costs ideas, not statistical honesty).
* **Book** — SINGLE-mode marginal value is scored against the controller's
  Pareto **archive** (the locked design decision), which evolves as the run
  progresses.
* **Persistence** — the controller state (population, archive, lineage,
  N_trials) is checkpointed to the scope's ``evolution/`` folder every
  generation; the final archive is materialised into real factor files and
  persisted to the scope's factor DB at the end, so downstream comparison /
  fund runs consume an evolution prerun exactly like a oneshot one.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from quant_fund_agent.agents.factor_research.evolution.controller import (
    ControllerConfig,
    EvaluatedGenome,
    EvolutionController,
)
from quant_fund_agent.agents.factor_research.evolution.genome import (
    FactorProgram,
    Genome,
)
from quant_fund_agent.agents.factor_research.evolution.mutation import (
    build_crossover_prompt,
    build_mutation_prompt,
    build_refine_prompt,
    jitter_variants,
    parse_child_response,
    random_jitter_child,
)
from quant_fund_agent.agents.factor_research.evolution.reflection import mutation_brief
from quant_fund_agent.research_eval.fitness import FitnessResult

# LLM cost metering + per-role attribution (defensive: no-op fallbacks so the
# loop keeps working against an older quant_fund_agent.llm without the meter).
try:
    from quant_fund_agent.llm import set_llm_role, usage_summary
except ImportError:  # pragma: no cover — older llm.py without the meter
    import contextlib

    def usage_summary() -> dict:
        return {}

    @contextlib.contextmanager
    def set_llm_role(role: str):
        yield

log = logging.getLogger("evolution.loop")


@dataclass
class EvolutionRunConfig:
    """Every knob of one evolutionary research run (serialised for the record)."""

    # ── search shape ──
    # "evolve" (default): the full evolutionary search — NSGA-II tournament
    # parent selection, semantic mutation, crossover, jitter, migration.
    # "refine": the non-evolutionary ablation — every seeded factor is refined
    # by the LLM against its own deterministic evaluation report (same factor,
    # same mechanism, at most ``refine_rounds`` times), demes continuously
    # re-seed fresh graphrag-grounded ideas when they run out of refinement
    # work, and the only combination operator is the occasional explicit
    # cross-group synthesis (``p_cross_group``).  No tournament selection, no
    # same-group crossover, no jitter mutation, no migration; scoring, gates,
    # Pareto archives, progressive reveal, curation and publish deflation are
    # identical to "evolve".
    variant: str = "evolve"            # "evolve" | "refine"
    refine_rounds: int = 2             # max LLM refinements per lineage (refine)
    generations: int = 5
    population_size: int = 10          # per island
    children_per_generation: int = 8   # candidates proposed per generation
    children_per_deme: int | None = None  # grouped mode; None preserves legacy total
    # Concurrent LLM proposal calls per generation (1 = fully sequential,
    # byte-identical legacy behaviour). Children within a generation are
    # independent given generation-start state: RNG draws and parent selection
    # stay sequential in the main thread, only the LLM round-trips (and the
    # per-idea seeding debate/codegen calls) fan out; admissions/evaluation
    # stay sequential in child-index order so scoring semantics are unchanged.
    llm_workers: int = 1
    n_mechanism_groups: int = 1
    # "exact": the graph must yield exactly n_mechanism_groups usable groups (raise
    # otherwise — legacy behaviour).  "max": n_mechanism_groups is a HARD UPPER
    # LIMIT; the run uses however many usable groups the knowledge graph actually
    # forms (>=1), and the config is shrunk to the resolved count before the
    # controller is built.
    mechanism_groups_mode: str = "exact"
    demes_per_group: int = 1
    # Legacy single-layer alias.  If set above 1 while grouped fields are left at
    # defaults, it is interpreted as demes_per_group.
    n_islands: int = 1
    migration_every: int = 5
    seed: int = 0
    unit: str = "single"               # "single" | "set"
    set_size: int = 3                  # initial members per SET genome
    # Optional hard cap on each mechanism group's Pareto archive (threaded into
    # ControllerConfig).  None (default) keeps the unbounded behaviour.
    archive_cap_per_group: int | None = None

    # ── operator mix (probabilities; renormalised) ──
    p_llm_semantic: float = 0.6
    p_crossover: float = 0.25
    p_jitter: float = 0.15
    p_cross_group: float = 0.10
    # With this probability an ``llm_semantic`` child is generated in "creative
    # mode" (a clearly-marked prompt section asking for a genuinely novel
    # mechanism, not derived from the parent or any paper).  At seeding,
    # ``round(creative_frac * n_ideas)`` ideas per group come from the
    # knowledge-only brainstorm path with the same novelty section appended.
    # 0.0 (default) is byte-identical to the historical behaviour.
    creative_frac: float = 0.0

    # ── seeding (generation 0, via the existing brainstorm/codegen path) ──
    n_seed_ideas: int = 8
    seed_ideas_per_group: int | None = None
    seed_papers: int = 0               # papers pulled for the seed brainstorm (0 → knowledge-only)

    # ── retrieval (P2): grounds the seed brainstorm in retrieved papers ──
    retrieval: str = "none"            # "none" | "rag"  ("graphrag": P4)
    retrieval_cardinality: str = "1toN"  # "1toN" | "Nto1" | "NtoM"
    rag_k: int = 4                     # papers retrieved per brainstorm

    # ── agent split + debate (P3) ──
    # "on" splits the LLM-semantic operator into Hypothesis → Debate → Codegen
    # (per-role models via {HYPOTHESIS,DEBATE,CODEGEN}_LLM_MODEL) and debates
    # seed / crossover ideas before they cost an evaluation.  "off" keeps the
    # single-call P1 operator — the ablation arm.
    debate: str = "off"                # "on" | "off"

    # ── evaluation (threads straight into research_eval) ──
    target_horizon: int = 6
    is_frac: float = 0.6
    val_frac: float = 0.2
    stability_blocks: int = 4
    plateau_scales: tuple[float, ...] = (0.9, 1.1)
    cutoff_date: str | None = None
    force_prediction_horizon: bool = True

    # ── progressive data reveal (default OFF → byte-identical to today) ──
    # When on, the dev window is revealed block-by-block across generations (an
    # expanding IS + sliding VAL), so each generation is partly scored on data no
    # earlier selection saw.  ``is_frac``/``val_frac`` are ignored in this mode; the
    # split is driven by the per-generation calendar schedule instead.  The final
    # ``test_frac`` tail is never revealed (unchanged TEST semantics).
    progressive: bool = False
    test_frac: float = 0.2       # final never-revealed TEST tail (progressive mode)
    seed_frac: float = 0.45      # fraction of DEV visible at generation 0
    reveal_every: int = 1        # generations between reveals
    val_blocks: int = 2          # sliding VAL = last val_blocks blocks
    # --final-holdout: keep one extra dev block hidden from EVERY generation and
    # reveal it only in a terminal rescore-only step after the last generation,
    # so the final front is ranked on data no selection pressure ever queried.
    final_holdout: bool = False
    # Two-phase walk-forward schedule (2026-08-01): when wf_blocks > 0 the LAST
    # wf_blocks generations each reveal one fixed wf_block_bars-bar block at the
    # dev tail (prequentially scored before the archive may adapt to it); the
    # earlier generations reveal [seed_end, wf_start) in equal blocks at the
    # reveal_every cadence.  Incompatible with final_holdout.
    wf_blocks: int = 0
    wf_block_bars: int = 126

    # --graph-readonly: never write factor provenance back into the knowledge
    # graph (comparison ladders need every arm to rank mechanism groups from the
    # identical graph snapshot; the ranking is factor-coverage-based).
    graph_readonly: bool = False

    # ── fitness: four Pareto axes ──
    independence_metric: str = "residual_ic"   # "residual_ic" | "delta_participation"
    # nonlinear LOCO combiner so conditioning/interaction value (e.g. a volatility
    # state variable) scores a positive marginal value; "ridge" = additive-only.
    marginal_model: str = "gradient_boosting"

    # ── economic realism (P5, WS3) — folded in, no new axis; all default OFF so the
    # baseline arm is unchanged, enabled as an ablation dimension ──
    gate_turnover: float | None = None         # cost_ok gate floor (None = off)
    cost_rate: float = 5e-4                     # net-of-cost DIAG cost per unit turnover
    perturbation_weight: float = 0.0           # marginal-axis perturbation probe weight (0 = off)
    perturbation_sigma: float = 0.5            # perturbation shock stdev (z-units)

    # ── two-stage curation (Lever 2): keep every gate-passer, curate once ──
    # "archive": the one-stage default — persist the Pareto archive (old behaviour).
    # "greedy"/"elastic_net": persist the curated kept-pool instead, so the final
    # book is not limited to the domination-pruned front.
    curation: str = "archive"
    n_keep: int | None = None                  # target book size (None → auto-sized)

    # ── selection-time deflation (WS1): the multiple-testing honesty control, moved
    # off the per-candidate search gate onto the final published book.  "off"
    # (default) = discovery mode (keep everything, deflation reported not enforced);
    # "on" = validation mode (narrow the book, pruning by marginal contribution, to
    # what beats selection luck for the run's N_trials). ──
    selection_deflation: str = "off"

    # ── data ──
    data_dir: str = "ticker_data"
    n_tickers: int | None = 15
    fixed_book: list[dict[str, Any]] = field(default_factory=list)
    # reference "factor zoo" (e.g. the ~86 base factors) for the novelty DIAG (WS4);
    # {"factor_id","code"} dicts.  Empty = novelty diagnostic off.
    reference_book: list[dict[str, Any]] = field(default_factory=list)

    # ── cross-run experience memory (P6, WS5) — per-config, opt-in ──
    memory: bool = False               # accumulate survivors + attempt tallies across runs
    memory_config: str | None = None   # config name → memory file path (per-config keyed)
    # Per-arm memory keying: when set, the experience store is keyed by this
    # string instead of the config-derived key (so ablation arms don't share
    # one memory).  None (default) keeps the config-derived behaviour.
    memory_key: str | None = None

    # ── output ──
    out_dir: str = "data/evolution"    # overridden by the entrypoint to the scope dir

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["plateau_scales"] = list(self.plateau_scales)
        return d

    @property
    def effective_demes_per_group(self) -> int:
        if self.demes_per_group == 1 and self.n_islands != 1:
            return max(1, self.n_islands)
        return max(1, self.demes_per_group)

    @property
    def effective_seed_ideas_per_group(self) -> int:
        return max(1, self.seed_ideas_per_group or self.n_seed_ideas)


def resolve_mechanism_groups(
    cfg: EvolutionRunConfig,
    fields: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """Resolve the upper knowledge-graph layer once, before any seed generation."""
    n_groups = max(1, cfg.n_mechanism_groups)
    if n_groups == 1 and cfg.retrieval != "graphrag":
        return [{"mechanism_group_id": 0, "community_id": None,
                 "focus": "", "mechanisms": []}]
    if cfg.retrieval != "graphrag":
        raise ValueError(
            "multiple mechanism groups require retrieval='graphrag'; the upper "
            "layer must be grounded in the knowledge graph")

    from quant_fund_agent.knowledge.graph_query import mechanism_group_specs
    from quant_fund_agent.knowledge.graph_store import KnowledgeGraph

    graph = KnowledgeGraph.load()
    specs = mechanism_group_specs(graph, n_groups, list(fields or []))
    if getattr(cfg, "mechanism_groups_mode", "exact") == "max":
        # n_mechanism_groups is an upper limit: keep every usable (focused)
        # group the graph formed, renumbered contiguously.
        usable = [s for s in specs if s.get("focus")][:n_groups]
        if not usable:
            raise ValueError(
                "knowledge graph formed no usable mechanism groups "
                "(is the graph built and non-trivial?)")
        for k, spec in enumerate(usable):
            spec["mechanism_group_id"] = k
        if len(usable) < n_groups:
            log.info("mechanism_groups_mode=max: graph yields %d usable "
                     "group(s) (upper limit %d)", len(usable), n_groups)
        return usable
    if len(specs) != n_groups or any(not s.get("focus") for s in specs):
        raise ValueError(
            f"knowledge graph could form only {len(specs)} usable mechanism groups "
            f"for requested n_mechanism_groups={n_groups}")
    return specs


# ── LLM plumbing (patchable in tests) ─────────────────────────────────────────

def _get_llm(temperature: float, role: str | None = None):
    """Per-role chat LLM (P3): ``role`` reads ``{ROLE}_LLM_MODEL`` / ``_PROVIDER``
    (e.g. ``HYPOTHESIS_LLM_MODEL``) and falls back to the run's research model —
    so tokens are spent where they matter (`--hypothesis-model` etc.)."""
    import os

    from quant_fund_agent.llm import make_chat_llm

    model = provider = None
    if role:
        model = os.getenv(f"{role.upper()}_LLM_MODEL") or None
        provider = os.getenv(f"{role.upper()}_LLM_PROVIDER") or None
    return make_chat_llm(model=model, provider=provider, temperature=temperature,
                         timeout=120, max_retries=4)


def _invoke(llm: Any, prompt: str) -> str:
    resp = llm.invoke(prompt)
    return getattr(resp, "content", str(resp))


# ── seeding via the existing brainstorm/codegen path ──────────────────────────

def _parallel_map(fn: Callable[[Any], Any], items: Sequence[Any],
                  workers: int) -> list[Any]:
    """Order-preserving map with a thread pool (``workers`` ≤ 1 = plain loop).

    Used for independent LLM round-trips (seeding debate/codegen, per-child
    proposals): items must not mutate shared loop state — RNG draws and
    admissions stay in the caller's thread.
    """
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as ex:
        return list(ex.map(fn, items))


def _codegen_program(
    llm: Any,
    idea: Any,
    data_context: str,
    expected_prediction_horizon: int | None = None,
) -> FactorProgram | None:
    """One idea → validated (in-memory) FactorProgram, with one feedback retry.

    Mirrors the oneshot ``generate_code`` node but compiles in-memory instead of
    materialising — evolution candidates only become files if they survive.
    """
    from quant_fund_agent.agents.factor_research.graph import (
        _codegen_prompt,
        _parse_json,
    )
    from quant_fund_agent.factors.inmem import compile_factor

    feedback = ""
    for _attempt in range(2):
        try:
            content = _invoke(llm, _codegen_prompt(idea, data_context, feedback=feedback))
            code = _parse_json(content).get("code", "")
            compile_factor(
                code, idea.factor_id, smoke=True,
                expected_prediction_horizon=expected_prediction_horizon)
            return FactorProgram(
                factor_id=idea.factor_id,
                code=code,
                name=idea.name,
                category=idea.category,
                trading_idea=idea.trading_idea,
                description=idea.description,
                prediction_horizon=idea.prediction_horizon,
                suggested_horizons=list(idea.suggested_horizons or []),
                expected_sign=getattr(idea, "expected_sign", None),
                source_paper_ids=list(idea.source_paper_ids or []),
            )
        except Exception as e:  # noqa: BLE001 — feed the error back once, then drop
            feedback = str(e)
            log.info("[seed:%s] codegen attempt failed (%s)", idea.factor_id, e)
    return None


def seed_programs(cfg: EvolutionRunConfig, data_context: str,
                  existing_ids: set[str],
                  fields: Sequence[str] | None = None,
                  mechanism_group: dict[str, Any] | None = None,
                  ) -> list[FactorProgram]:
    """Generation 0: brainstorm ideas (GraphRAG / RAG / papers / knowledge-only)
    → validated programs."""
    from quant_fund_agent.agents.factor_research.graph import _brainstorm_one
    from quant_fund_agent.agents.factor_research.state import FactorIdea, PaperSnippet
    from quant_fund_agent.mcp import research_client

    group_id = int((mechanism_group or {}).get("mechanism_group_id", 0))
    session_id = f"evo-seed-{cfg.seed}-group-{group_id}"
    brainstorm_llm = _get_llm(temperature=0.7, role="hypothesis")
    known = set(existing_ids)
    raw_ideas: list[dict] = []
    graph = None           # GraphRAG: set when the knowledge graph is in play
    mech_by_fid: dict[str, str] = {}

    # --creative-frac: reserve a slice of the seed ideas for the knowledge-only
    # brainstorm path with the novelty section appended (0 → byte-identical).
    n_creative = (int(round(cfg.creative_frac * cfg.n_seed_ideas))
                  if cfg.creative_frac > 0 else 0)
    n_regular = max(0, cfg.n_seed_ideas - n_creative)

    if cfg.retrieval in ("rag", "graphrag"):
        # Retrieval-grounded seeding: embedding retrieval picks the papers
        # (date-gated) and the cardinality mode shapes the call(s); citations
        # come back verified.  GraphRAG additionally steers the query toward
        # computable gap mechanisms / under-covered communities and asks each
        # idea to name the mechanism it exploits (provenance link-back).
        from quant_fund_agent.knowledge.embed_store import EmbedStore
        from quant_fund_agent.knowledge.retrieval import retrieve_and_brainstorm

        focus = str((mechanism_group or {}).get("focus") or "") or None
        gap_names: list[str] | None = None
        if cfg.retrieval == "graphrag":
            from quant_fund_agent.knowledge.graph_query import (
                computable_unexploited,
            )
            from quant_fund_agent.knowledge.graph_store import (
                DEFAULT_GRAPH_PATH,
                KnowledgeGraph,
            )

            try:
                graph = KnowledgeGraph.load()
                if mechanism_group is not None:
                    gap_names = list(mechanism_group.get("mechanisms") or []) or None
                else:
                    gap_ids = computable_unexploited(graph, list(fields or []))
                    gap_names = [graph.g.nodes[m].get("name", m)
                                 for m in gap_ids][:8] or None
                log.info("graphrag group %d seeding: focus=%r mechanisms=%s",
                         group_id, (focus or "")[:120], gap_names or [])
            except FileNotFoundError:
                log.warning("knowledge graph missing (%s) — run "
                            "scripts/build_knowledge_graph.py first; falling "
                            "back to flat RAG", DEFAULT_GRAPH_PATH)
                graph = None

        store = EmbedStore()
        # Scope-filtered retrieval: when the run's field scope carries no
        # fundamental / estimate / event field, restrict retrieval to papers
        # harvested under price/general data scopes.
        allowed_scopes = None
        try:
            from quant_fund_agent.data.tiers import TIERS

            non_price = (TIERS.get("fundamental", frozenset())
                         | TIERS.get("estimates", frozenset())
                         | TIERS.get("events", frozenset()))
            if fields is not None and not any(f in non_price for f in fields):
                allowed_scopes = {"price", "general"}
        except Exception as e:  # noqa: BLE001 — never let scoping break seeding
            log.warning("could not resolve retrieval data scopes (%s)", e)
        rb_kwargs: dict[str, Any] = dict(
            n_ideas=n_regular, known_ids=known,
            data_context=data_context,
            cardinality=cfg.retrieval_cardinality,
            k_papers=max(cfg.rag_k, cfg.seed_papers or 0) or 4,
            cutoff_date=cfg.cutoff_date,
            focus=focus, gaps=gap_names,
            mechanism_tags=gap_names if graph is not None else None,
        )
        if n_regular > 0:
            with set_llm_role("brainstorm"):
                try:
                    try:
                        raw_ideas = retrieve_and_brainstorm(
                            brainstorm_llm, store,
                            allowed_scopes=allowed_scopes, **rb_kwargs)
                    except TypeError:
                        # older retrieval signature without ``allowed_scopes``
                        raw_ideas = retrieve_and_brainstorm(
                            brainstorm_llm, store, **rb_kwargs)
                except Exception as e:  # noqa: BLE001 — a bad LLM reply costs
                    # this group's grounded ideas, never the whole run
                    log.warning("grounded seed brainstorm failed for group %s "
                                "(%s) — continuing without it", group_id, e)
                    raw_ideas = []
        mech_by_fid = {
            (raw.get("factor_id") or "").strip().lower(): str(raw["mechanism"])
            for raw in raw_ideas
            if raw.get("mechanism") and raw.get("factor_id")}
    else:
        papers: list[PaperSnippet] = []
        if cfg.seed_papers > 0:
            try:
                rows = research_client.load_papers(n=cfg.seed_papers,
                                                   cutoff_date=cfg.cutoff_date,
                                                   strategy="random")
                papers = [PaperSnippet(**row) for row in rows]
            except Exception as e:  # noqa: BLE001 — degrade to knowledge-only
                log.warning("seed paper load failed (%s) — knowledge-only brainstorm", e)

        def _safe_brainstorm(*args) -> list[dict]:
            """One bad LLM reply must never kill a long run — skip the call."""
            try:
                return _brainstorm_one(brainstorm_llm, *args)
            except Exception as e:  # noqa: BLE001
                log.warning("seed brainstorm call failed (%s) — skipping", e)
                return []

        with set_llm_role("brainstorm"):
            if papers and n_regular > 0:
                per_paper = max(1, -(-n_regular // len(papers)))  # ceil division
                for paper in papers:
                    if len(raw_ideas) >= n_regular:
                        break
                    raw_ideas.extend(_safe_brainstorm(
                        paper, per_paper, known, session_id, data_context))
            elif n_regular > 0:
                raw_ideas = _safe_brainstorm(
                    None, n_regular, known, session_id, data_context)

    if n_creative > 0:
        # Creative slice: knowledge-only brainstorm with the novelty section.
        from quant_fund_agent.agents.factor_research.evolution.mutation import (
            CREATIVE_NOVELTY_SECTION,
        )

        creative_context = f"{data_context}\n\n{CREATIVE_NOVELTY_SECTION}"
        with set_llm_role("brainstorm"):
            try:
                creative_ideas = _brainstorm_one(
                    brainstorm_llm, None, n_creative, known,
                    f"{session_id}-creative", creative_context)
            except Exception as e:  # noqa: BLE001
                log.warning("creative seed brainstorm failed (%s) — skipping", e)
                creative_ideas = []
        raw_ideas = list(raw_ideas) + creative_ideas

    ideas: list[FactorIdea] = []
    seen: set[str] = set()
    for raw in raw_ideas:
        if len(ideas) >= cfg.n_seed_ideas:
            break
        idea = coerce_idea(
            raw, cfg.target_horizon, force_horizon=cfg.force_prediction_horizon)
        if idea is None or idea.factor_id in seen or idea.factor_id in known:
            continue
        seen.add(idea.factor_id)
        known.add(idea.factor_id)
        ideas.append(idea)

    # Debate (P3): weak seed hypotheses are challenged BEFORE codegen spends
    # tokens implementing them; revisions replace the original idea.
    if cfg.debate == "on" and ideas:
        from quant_fund_agent.agents.factor_research.debate import run_debate

        debate_llm = _get_llm(temperature=0.3, role="debate")

        def _debate_one(idea: FactorIdea) -> FactorIdea | None:
            from quant_fund_agent.llm import budget_exhausted

            if budget_exhausted():
                return None
            with set_llm_role("debate"):
                verdict, final, _ = run_debate(
                    debate_llm, _idea_payload(idea), data_context=data_context,
                    book=[])
            if verdict == "reject":
                log.info("[seed:%s] rejected in debate", idea.factor_id)
                return None
            return _apply_idea_revision(
                idea, final, cfg.target_horizon,
                force_horizon=cfg.force_prediction_horizon)

        surviving = [i for i in _parallel_map(_debate_one, ideas,
                                              cfg.llm_workers) if i is not None]
        log.info("debate kept %d/%d seed idea(s)", len(surviving), len(ideas))
        ideas = surviving

    codegen_llm = _get_llm(temperature=0.2, role="codegen")
    fixed_horizon = cfg.target_horizon if cfg.force_prediction_horizon else None

    def _codegen_one(idea: FactorIdea) -> FactorProgram | None:
        from quant_fund_agent.llm import budget_exhausted

        if budget_exhausted():
            return None
        with set_llm_role("codegen"):
            return _codegen_program(
                codegen_llm, idea, data_context,
                expected_prediction_horizon=fixed_horizon)

    programs = []
    for prog in _parallel_map(_codegen_one, ideas, cfg.llm_workers):
        if prog is not None:
            prog.mechanism_group_id = group_id
            prog.mechanism = mech_by_fid.get(prog.factor_id, "")
            programs.append(prog)
    log.info("seeded %d/%d program(s) from %d idea(s)",
             len(programs), cfg.n_seed_ideas, len(ideas))

    # GraphRAG provenance link-back: Paper → Mechanism → Factor → Field.
    if graph is not None and mech_by_fid:
        link_programs_into_graph(graph, programs, mech_by_fid,
                                 readonly=cfg.graph_readonly)
    return programs


def link_programs_into_graph(graph, programs, mech_by_fid, *,
                             readonly: bool = False) -> None:
    """Provenance link-back of seeded programs (Paper → Mechanism → Factor → Field).

    ``readonly=True`` (``--graph-readonly``) is a strict no-op — neither the
    in-memory graph nor ``graph.json`` is touched, so every arm of a comparison
    ladder resolves its mechanism groups from the identical graph snapshot
    (the group ranking is coverage-based and factor nodes shift it).
    """
    if readonly:
        log.info("graph read-only: skipping provenance link-back of %d program(s)",
                 len(programs))
        return
    from quant_fund_agent.knowledge.empirical_edges import (
        link_factor_to_mechanism,
        refresh_field_usage,
    )
    from quant_fund_agent.factors.inmem import compile_factor

    inputs_by_fid: dict[str, list[str]] = {}
    for prog in programs:
        mech = mech_by_fid.get(prog.factor_id)
        if mech:
            link_factor_to_mechanism(graph, prog.factor_id, mech)
        try:
            cls = compile_factor(prog.code, prog.factor_id)
            inputs_by_fid[prog.factor_id] = list(
                getattr(cls, "inputs", None) or ["close"])
        except Exception:  # noqa: BLE001 — provenance is best-effort
            pass
    if inputs_by_fid:
        refresh_field_usage(graph, inputs_by_fid)
    try:
        graph.save()
    except Exception as e:  # noqa: BLE001
        log.warning("could not save knowledge graph: %s", e)


def coerce_idea(
    raw: dict[str, Any],
    default_horizon: int,
    *,
    force_horizon: bool = False,
) -> "Any | None":
    """Raw idea dict (brainstorm / hypothesis / debate revision) → FactorIdea."""
    from quant_fund_agent.agents.factor_research.graph import (
        _coerce_horizon,
        _coerce_horizon_list,
    )
    from quant_fund_agent.agents.factor_research.state import FactorIdea

    fid = (raw.get("factor_id") or "").strip().lower()
    if not fid:
        return None
    sign_raw = raw.get("expected_sign")
    try:
        sign = int(np.sign(int(sign_raw))) or None if sign_raw is not None else None
    except (TypeError, ValueError):
        sign = None
    horizon = (
        int(default_horizon) if force_horizon
        else _coerce_horizon(raw.get("prediction_horizon"), default_horizon)
    )
    return FactorIdea(
        factor_id=fid,
        name=raw.get("name", fid),
        category=raw.get("category", "other"),
        trading_idea=raw.get("trading_idea", ""),
        description=raw.get("description", ""),
        prediction_horizon=horizon,
        suggested_horizons=(
            [horizon] if force_horizon
            else _coerce_horizon_list(raw.get("suggested_horizons"))
        ),
        expected_sign=sign,
        source_paper_ids=list(raw.get("source_paper_ids", []) or []),
    )


def _idea_payload(idea: Any) -> dict[str, Any]:
    """The hypothesis dict the debate sees for a FactorIdea / FactorProgram."""
    return {
        "factor_id": idea.factor_id,
        "name": getattr(idea, "name", idea.factor_id),
        "category": getattr(idea, "category", "other"),
        "trading_idea": getattr(idea, "trading_idea", ""),
        "description": getattr(idea, "description", ""),
        "prediction_horizon": getattr(idea, "prediction_horizon", 6),
        "suggested_horizons": list(getattr(idea, "suggested_horizons", []) or []),
        "expected_sign": getattr(idea, "expected_sign", None),
        "source_paper_ids": list(getattr(idea, "source_paper_ids", []) or []),
    }


def _apply_idea_revision(
    idea: Any,
    final: dict[str, Any],
    default_horizon: int,
    *,
    force_horizon: bool = False,
) -> Any:
    """Fold a debate revision back into the idea (keeping the id if unchanged)."""
    revised = coerce_idea(final, default_horizon, force_horizon=force_horizon)
    if revised is None:
        return idea
    if not revised.trading_idea:
        revised.trading_idea = idea.trading_idea
    if not revised.source_paper_ids:
        revised.source_paper_ids = idea.source_paper_ids
    return revised


# ── the run ───────────────────────────────────────────────────────────────────

class EvolutionLoop:
    """Drives one evolutionary research run (the deterministic controller +
    the LLM operators + the MCP evaluation seam)."""

    def __init__(self, cfg: EvolutionRunConfig,
                 data_context: str | None = None,
                 fields: list[str] | None = None) -> None:
        self.cfg = cfg
        # "max" group mode: consult the knowledge graph BEFORE the controller is
        # built and shrink the config to the resolved group count, so the island
        # structure, cross-group operators and seeding all agree with the graph.
        self._resolved_groups: list[dict[str, Any]] | None = None
        if (getattr(cfg, "mechanism_groups_mode", "exact") == "max"
                and cfg.n_mechanism_groups > 1):
            self._resolved_groups = resolve_mechanism_groups(cfg, fields)
            if len(self._resolved_groups) != cfg.n_mechanism_groups:
                cfg = replace(cfg,
                              n_mechanism_groups=len(self._resolved_groups))
                self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        demes_per_group = cfg.effective_demes_per_group
        controller_kwargs: dict[str, Any] = dict(
            population_size=cfg.population_size,
            n_mechanism_groups=cfg.n_mechanism_groups,
            demes_per_group=demes_per_group,
            migration_every=cfg.migration_every,
            seed=cfg.seed,
        )
        try:
            self.controller = EvolutionController(ControllerConfig(
                archive_cap_per_group=cfg.archive_cap_per_group,
                **controller_kwargs))
        except TypeError:
            # older ControllerConfig without the cap field
            self.controller = EvolutionController(
                ControllerConfig(**controller_kwargs))
        # Archive-eviction events flow into the run lineage (the same list
        # _checkpoint writes to lineage.jsonl); guarded so an older controller
        # without the sink still works.
        if hasattr(self.controller, "event_sink"):
            self.controller.event_sink = self._lineage_event
        self.mechanism_groups: list[dict[str, Any]] = self._resolved_groups or [
            {"mechanism_group_id": g, "community_id": None,
             "focus": "", "mechanisms": []}
            for g in range(max(1, cfg.n_mechanism_groups))
        ]
        self.briefs: dict[str, str] = {}      # genome_id → reflection brief
        self.fields = fields                   # run-constant field set (panel cache key)
        self.data_context = data_context or self._build_data_context()
        self.out_dir = Path(cfg.out_dir)
        self.known_ids: set[str] = set()
        # Parallel proposal support: guards factor-id allocation, and holds ids
        # handed out this generation but not yet admitted (so two concurrent
        # children can never be given the same id).
        self._id_lock = threading.Lock()
        self._pending_ids: set[str] = set()
        self.failures: list[dict[str, Any]] = []

        # ── experience memory (WS5): load the per-config graph, read the exhausted /
        # summary steering signals up front (they reflect PRIOR runs) ──
        self.memory = None
        self._memory_path = None
        self._exhausted_mechs: list[str] = []
        self._memory_brief: str = ""
        self._mech_by_genome: dict[str, str] = {}   # genome_id → mechanism (child inherit)
        if cfg.memory:
            from quant_fund_agent.knowledge import experience
            # Per-arm keying: an explicit memory_key wins over the config-derived
            # key so ablation arms can keep separate experience stores.
            self._memory_path = experience.memory_graph_path(
                cfg.memory_key or cfg.memory_config or "default")
            self.memory = experience.load_or_new(self._memory_path)
            self._exhausted_mechs = experience.exhausted_mechanisms(self.memory)
            self._memory_brief = experience.memory_summary(self.memory)
            log.info("experience memory: %s (exhausted: %s)",
                     self._memory_path, self._exhausted_mechs or "none")
            # READ seam: steer seeding/codegen away from exhausted mechanisms.
            if self._exhausted_mechs:
                self.data_context = (
                    self.data_context
                    + "\n\nEXPERIENCE MEMORY — AVOID these exhausted mechanisms (many "
                      "prior attempts, little OOS edge); prefer novel angles: "
                    + ", ".join(self._exhausted_mechs) + ".")
        self._llms: dict[str, Any] = {}    # role → chat model (built lazily)
        self._program_pool: dict[str, FactorProgram] = {}  # every program ever admitted
        self.fixed_book = self._dedupe_fixed_book(cfg.fixed_book)
        # progressive reveal (default OFF): the per-generation frontier schedule
        self._schedule: list[Any] | None = None
        self._window: Any | None = None
        # resume-dedup sets for the appended-to run artifacts (lazily loaded from
        # the existing files, so a resumed run never duplicates a row)
        self._prequential_done: set[int] | None = None
        self._gen_quality_done: set[int] | None = None
        # refinement variant: island → ordered lineage entries, each the latest
        # version of one factor lineage plus how many refinements it has used
        # ({"eg": EvaluatedGenome, "refines": int, "last_gen": int}).
        if cfg.variant not in ("evolve", "refine"):
            raise ValueError(f"unknown variant {cfg.variant!r}")
        if cfg.variant == "refine" and cfg.unit != "single":
            raise ValueError("variant='refine' supports unit='single' only")
        self._refine_entries: dict[int, list[dict[str, Any]]] = {}

    def _lineage_event(self, event: dict) -> None:
        """Controller event sink: archive events land in the run lineage.

        No timestamp is added — the lineage stays deterministic given the seed.
        """
        self.controller.lineage.append(dict(event))

    @staticmethod
    def _jsonl_generations(path: Path) -> set[int]:
        """The ``generation`` values already recorded in an appended jsonl file."""
        done: set[int] = set()
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    g = json.loads(line).get("generation")
                except (json.JSONDecodeError, AttributeError):
                    continue
                if g is not None:
                    done.add(int(g))
        return done

    @staticmethod
    def _dedupe_fixed_book(book: Sequence[dict[str, Any]] | None) -> list[dict[str, Any]]:
        """A run-constant conditioning book, never inserted into the archive."""
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for row in book or []:
            fid = str(row.get("factor_id") or "").strip()
            code = str(row.get("code") or "")
            if not fid or not code or fid in seen:
                continue
            seen.add(fid)
            out.append({**row, "factor_id": fid, "code": code})
        return out

    def _role_llm(self, role: str, temperature: float) -> Any:
        if role not in self._llms:
            self._llms[role] = _get_llm(temperature=temperature, role=role)
        return self._llms[role]

    # ── context helpers ──

    def _build_data_context(self) -> str:
        from quant_fund_agent.agents.factor_research.prompts import build_data_context
        from quant_fund_agent.pipeline import _infer_seconds_per_bar

        allowed = self.fields
        if allowed is None:
            try:
                from quant_fund_agent.data import usable_fields

                allowed = sorted(usable_fields())
                self.fields = allowed
            except Exception as e:  # noqa: BLE001
                log.warning("could not resolve usable fields (%s) — un-gated", e)
        spb = None
        try:
            spb = _infer_seconds_per_bar(self.cfg.data_dir)
        except Exception:  # noqa: BLE001
            pass
        fixed = self.cfg.target_horizon if self.cfg.force_prediction_horizon else None
        return build_data_context(allowed, spb, fixed_prediction_horizon=fixed)

    def _load_known_ids(self) -> None:
        from quant_fund_agent.mcp import research_client

        try:
            self.known_ids = set(research_client.existing_factor_ids(scope="package"))
        except Exception as e:  # noqa: BLE001
            log.warning("could not load existing factor ids (%s)", e)
            self.known_ids = set()
        self.known_ids.update(b["factor_id"] for b in self.fixed_book)

    def _unique_id(self, base: str) -> str:
        base = (base or "factor").strip().lower()[:48] or "factor"
        with self._id_lock:
            fid = base
            k = 1
            taken = self.known_ids | self._pending_ids | {
                p.factor_id for eg in self.controller.population()
                for p in eg.genome.programs}
            while fid in taken:
                fid = f"{base}_{k}"
                k += 1
            if self.cfg.llm_workers > 1:
                # reserve so a concurrent sibling can't be handed the same id
                self._pending_ids.add(fid)
            return fid

    def _group_context(self, mechanism_group_id: int) -> str:
        spec = self.mechanism_groups[
            mechanism_group_id % len(self.mechanism_groups)]
        focus = str(spec.get("focus") or "").strip()
        if not focus:
            return self.data_context
        return (
            self.data_context
            + "\n\nKNOWLEDGE-GRAPH MECHANISM GROUP\n"
            + "Develop this lineage within the following economic mechanism "
              "community. You may combine related mechanisms, but keep a clear "
              "causal link to the group:\n"
            + focus
        )

    # ── evaluation ──

    @property
    def _window_bounds(self) -> tuple[str | None, str | None]:
        """The (is_end, val_end) calendar bounds of the current progressive window,
        or (None, None) in legacy mode (→ the harness uses is_frac/val_frac)."""
        if self.cfg.progressive and self._window is not None:
            return self._window.is_end_ts, self._window.val_end_ts
        return None, None

    def _stamp_window(self, fitness: FitnessResult) -> FitnessResult:
        """Record which data window scored this fitness (progressive-reveal diag)."""
        if self.cfg.progressive and self._window is not None:
            fitness.diagnostics["scored_through"] = self._window.val_end_ts
            fitness.diagnostics["window_generation"] = self._window.generation
        return fitness

    def _book_for(self, program: FactorProgram,
                  archive_programs: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
        """LOCO book = fixed book + archive members, excluding the program itself."""
        book = [
            {"factor_id": b["factor_id"], "code": b["code"]}
            for b in self.fixed_book
            if b["factor_id"] != program.factor_id
        ]
        fixed_ids = {b["factor_id"] for b in book}
        book.extend(
            {"factor_id": fid, "code": code}
            for fid, code in archive_programs
            if fid != program.factor_id and fid not in fixed_ids
        )
        return book

    def _score_program(self, program: FactorProgram, book: list[dict[str, Any]],
                       n_trials: int, *, is_end: str | None,
                       val_end: str | None, include_probes: bool = True,
                       include_reference: bool = True) -> dict[str, Any]:
        """The shared client call for SINGLE scoring (billed eval AND re-score).

        ``include_probes=False`` / ``include_reference=False`` drop the jitter
        plateau probes and the reference-zoo diagnostics from the call — used by
        the archive rescore, where both were already computed at admission and
        re-executing every member's jittered code on the new window dominates the
        rescore's wall-clock (the admission-time plateau dock is carried forward
        by the caller instead).
        """
        from quant_fund_agent.mcp import research_client

        probes = ([{"factor_id": pid, "code": pcode}
                   for pid, pcode in jitter_variants(program, self.cfg.plateau_scales)]
                  if include_probes else [])
        reference = ([{"factor_id": r["factor_id"], "code": r["code"]}
                      for r in self.cfg.reference_book
                      if r.get("factor_id") != program.factor_id]
                     if include_reference else [])
        return research_client.evaluate_fitness(
            candidate={"factor_id": program.factor_id, "code": program.code,
                       "expected_sign": program.expected_sign},
            book=book,
            jitter=probes,
            reference=reference or None,
            target_horizon=self.cfg.target_horizon,
            is_frac=self.cfg.is_frac,
            val_frac=self.cfg.val_frac,
            is_end=is_end,
            val_end=val_end,
            n_trials=n_trials,
            stability_blocks=self.cfg.stability_blocks,
            cutoff_date=self.cfg.cutoff_date,
            data_dir=self.cfg.data_dir,
            n_tickers=self.cfg.n_tickers,
            fields=self.fields,
            independence_metric=self.cfg.independence_metric,
            marginal_model=self.cfg.marginal_model,
            gate_turnover=self.cfg.gate_turnover,
            cost_rate=self.cfg.cost_rate,
            perturbation_weight=self.cfg.perturbation_weight,
            perturbation_sigma=self.cfg.perturbation_sigma,
        )

    def _score_set(self, programs: Sequence[FactorProgram], candidate_id: str,
                   n_trials: int, *, is_end: str | None,
                   val_end: str | None) -> dict[str, Any]:
        """The shared client call for SET scoring (billed eval AND re-score)."""
        from quant_fund_agent.mcp import research_client

        return research_client.evaluate_set_fitness(
            [{"factor_id": p.factor_id, "code": p.code} for p in programs],
            target_horizon=self.cfg.target_horizon,
            is_frac=self.cfg.is_frac,
            val_frac=self.cfg.val_frac,
            is_end=is_end,
            val_end=val_end,
            n_trials=n_trials,
            stability_blocks=self.cfg.stability_blocks,
            cutoff_date=self.cfg.cutoff_date,
            data_dir=self.cfg.data_dir,
            n_tickers=self.cfg.n_tickers,
            fields=self.fields,
            candidate_id=candidate_id,
            marginal_model=self.cfg.marginal_model,
            gate_turnover=self.cfg.gate_turnover,
            cost_rate=self.cfg.cost_rate,
            perturbation_weight=self.cfg.perturbation_weight,
            perturbation_sigma=self.cfg.perturbation_sigma,
        )

    def evaluate_program(self, program: FactorProgram) -> FitnessResult | None:
        """Score one program via the MCP seam.  Returns None on eval failure."""
        is_end, val_end = self._window_bounds
        book = self._book_for(program, self.controller.archive_programs())
        res = self._score_program(
            program, book, self.controller.n_trials + 1,  # bill this look, commit on success
            is_end=is_end, val_end=val_end)
        if not res.get("ok"):
            log.info("[%s] evaluation failed: %s", program.factor_id, res.get("error"))
            self.failures.append({"factor_id": program.factor_id,
                                  "error": res.get("error")})
            return None
        self.controller.next_trial()
        return self._stamp_window(FitnessResult.from_dict(res["fitness"]))

    def evaluate_set(self, programs: Sequence[FactorProgram],
                     candidate_id: str) -> FitnessResult | None:
        """SET mode: score the whole member list jointly via the MCP seam."""
        is_end, val_end = self._window_bounds
        res = self._score_set(programs, candidate_id, self.controller.n_trials + 1,
                              is_end=is_end, val_end=val_end)
        if not res.get("ok"):
            log.info("[%s] set evaluation failed: %s", candidate_id, res.get("error"))
            self.failures.append({"factor_id": candidate_id,
                                  "error": res.get("error")})
            return None
        self.controller.next_trial()
        return self._stamp_window(FitnessResult.from_dict(res["fitness"]))

    # ── progressive data reveal (default OFF) ──

    def _init_progressive(self) -> None:
        """Fetch the run timeline and build the per-generation reveal schedule.

        Pure function of (config, cutoff-sliced index); the window in force is
        recomputed from ``controller.generation`` so a resumed run reproduces it.
        """
        from quant_fund_agent.agents.factor_research.evolution.progressive import (
            build_schedule,
        )
        from quant_fund_agent.mcp import research_client

        tl = research_client.panel_timeline(
            data_dir=self.cfg.data_dir, n_tickers=self.cfg.n_tickers,
            fields=self.fields, cutoff_date=self.cfg.cutoff_date)
        if not isinstance(tl, dict) or not tl.get("ok") or not tl.get("index"):
            raise RuntimeError(
                f"progressive reveal: could not fetch the panel timeline "
                f"({tl.get('error') if isinstance(tl, dict) else tl!r})")
        index = pd.to_datetime(list(tl["index"]))
        self._schedule = build_schedule(
            index, generations=self.cfg.generations, test_frac=self.cfg.test_frac,
            seed_frac=self.cfg.seed_frac, reveal_every=self.cfg.reveal_every,
            val_blocks=self.cfg.val_blocks, holdout_last=self.cfg.final_holdout,
            wf_blocks=self.cfg.wf_blocks, wf_block_bars=self.cfg.wf_block_bars)
        # The schedule addresses generations 0..G (+ a terminal G+1 window under
        # --final-holdout); the controller never advances past G.
        g = min(self.controller.generation, self.cfg.generations)
        self._window = self._schedule[g]
        # Resume corruption guard: the recomputed frontier must match the checkpoint.
        prog_path = self.out_dir / "progressive.json"
        if self.controller.generation > 0 and prog_path.exists():
            stored = json.loads(prog_path.read_text())
            if stored.get("frontier_ts") and \
                    stored["frontier_ts"] != self._window.val_end_ts:
                if self.cfg.final_holdout and \
                        stored["frontier_ts"] == self._schedule[-1].val_end_ts:
                    # The terminal holdout rescore already ran before this resume;
                    # restore its window (the rescore-only step is idempotent).
                    self._window = self._schedule[-1]
                else:
                    raise RuntimeError(
                        f"progressive-reveal resume mismatch at generation "
                        f"{self.controller.generation}: recomputed frontier "
                        f"{self._window.val_end_ts} != checkpointed "
                        f"{stored['frontier_ts']}.")
        log.info("progressive reveal: %d windows; generation %d frontier through %s",
                 len(self._schedule), self.controller.generation,
                 self._window.val_end_ts)

    def _prequential_score(self, gen: int, old_ts: str, new_ts: str) -> None:
        """Honest OOS score of the current archive on the block about to be revealed.

        ``[old_ts, new_ts)`` was never visible to any selection that produced this
        archive, so this is a genuine out-of-sample check.  One JSON line per reveal is
        appended to ``prequential.jsonl`` (a ``{"skipped": ...}`` row when the archive
        is empty or the window is too thin).  Validation only — nothing is gated on it.
        """
        from quant_fund_agent.mcp import research_client

        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Resume-dedup: one row per generation, ever (read the file once).
        if self._prequential_done is None:
            self._prequential_done = self._jsonl_generations(
                self.out_dir / "prequential.jsonl")
        if gen in self._prequential_done:
            log.info("prequential row for generation %d already recorded — skipped",
                     gen)
            return
        book = self.controller.archive_programs()
        if not book:
            row: dict[str, Any] = {"generation": gen, "start": old_ts, "end": new_ts,
                                   "skipped": "empty archive"}
        else:
            res = research_client.score_book_oos(
                [{"factor_id": fid, "code": code} for fid, code in book],
                start=old_ts, end=new_ts, target_horizon=self.cfg.target_horizon,
                data_dir=self.cfg.data_dir, n_tickers=self.cfg.n_tickers,
                fields=self.fields, marginal_model=self.cfg.marginal_model)
            if res.get("ok"):
                row = {"generation": gen, "start": old_ts, "end": new_ts,
                       "combined_oos_ic": res.get("combined_oos_ic"),
                       "n_obs": res.get("combined_n_obs"),
                       "per_factor_ic": res.get("per_factor_oos_ic"),
                       "pbo": (res.get("pbo") or {}).get("pbo"),
                       "archive_size": len(book),
                       "n_trials": self.controller.n_trials}
            else:
                row = {"generation": gen, "start": old_ts, "end": new_ts,
                       "skipped": res.get("error")}
        # Reveal stamps: which reveal this was and the frontier it advanced to.
        row["reveal_index"] = self._reveal_index(gen)
        row["frontier_ts"] = str(new_ts)
        if self._schedule is not None and gen < len(self._schedule):
            row["visible_end"] = self._schedule[gen].visible_end
        with (self.out_dir / "prequential.jsonl").open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        self._prequential_done.add(gen)

    def _reveal_index(self, gen: int) -> int | None:
        """Position of ``gen`` among this run's reveal generations (None if not one)."""
        if self._schedule is not None:
            # The schedule is the source of truth (required for the two-phase
            # walk-forward layout, identical to the cadence formula otherwise).
            gens = [w.generation for w in self._schedule if w.reveal]
            return gens.index(gen) if gen in gens else None
        from quant_fund_agent.agents.factor_research.evolution.progressive import (
            reveal_generations,
        )

        gens = reveal_generations(self.cfg.generations, max(1, self.cfg.reveal_every))
        if self.cfg.final_holdout and gen == self.cfg.generations + 1:
            return len(gens)   # the terminal holdout reveal follows the last in-run one
        return gens.index(gen) if gen in gens else None

    def _rescore_archive_on_window(self, gen: int) -> None:
        """Re-score every archive member on the just-advanced window (no billing).

        The LOCO book is a snapshot of the archive *before* re-scoring so the pass is
        order-independent; ``n_trials`` is passed unchanged (re-scores are not trials).
        Each member's drift is logged as a ``rescore`` lineage row (an overfitting
        diagnostic the thesis plots), then the controller re-prunes the archive.
        """
        archive = list(self.controller.archive)              # snapshot the member list
        snapshot = list(self.controller.archive_programs())  # snapshot the LOCO book
        is_end, val_end = self._window_bounds
        new_fitness: dict[str, FitnessResult] = {}
        for eg in archive:
            obj_before = eg.fitness.objective.to_dict()
            if eg.genome.unit == "set":
                res = self._score_set(eg.genome.programs, eg.genome.genome_id,
                                      self.controller.n_trials,  # NO +1 — not a new trial
                                      is_end=is_end, val_end=val_end)
            else:
                book = self._book_for(eg.genome.program, snapshot)
                res = self._score_program(eg.genome.program, book,
                                          self.controller.n_trials,
                                          is_end=is_end, val_end=val_end,
                                          include_probes=False,
                                          include_reference=False)
            if not res.get("ok"):
                # Visible failure (stale fitness kept — behaviour unchanged).
                self.controller.lineage.append({
                    "event": "rescore_failed", "generation": gen,
                    "genome_id": eg.genome.genome_id,
                    "error": res.get("error"),
                })
                log.warning("[%s] rescore failed on the new window: %s",
                            eg.genome.genome_id, res.get("error"))
                continue
            fitness = self._stamp_window(FitnessResult.from_dict(res["fitness"]))
            if eg.genome.unit != "set":
                # Jitter probes are skipped at rescore, so the plateau penalty is
                # not recomputed on the new window — carry the member's most recent
                # dock forward instead (admission-time, or an earlier carry), so a
                # knife-edge factor keeps its marginal-axis penalty for as long as
                # it lives in the archive.
                prev = eg.fitness.diagnostics or {}
                carried = prev.get("plateau_penalty")
                if carried is None:
                    carried = prev.get("plateau_penalty_carried")
                if carried and fitness.objective.marginal_value is not None:
                    fitness.objective.marginal_value = (
                        float(fitness.objective.marginal_value) - float(carried))
                    fitness.diagnostics["plateau_penalty_carried"] = float(carried)
            new_fitness[eg.genome.genome_id] = fitness
            diags = fitness.diagnostics or {}
            self.controller.lineage.append({
                "event": "rescore", "generation": gen,
                "genome_id": eg.genome.genome_id,
                "objective_before": obj_before,
                "objective_after": fitness.objective.to_dict(),
                "gates_after": fitness.gates.to_dict(),
                "reveal_index": self._reveal_index(gen),
                "scored_through": diags.get("scored_through"),
                "diagnostics": {
                    "val_ic": diags.get("val_ic"),
                    "marginal_value_raw": diags.get("marginal_value_raw"),
                    "coverage": diags.get("coverage"),
                },
            })
        self.controller.rescore_archive(new_fitness)

    def _advance_reveal(self, gen: int) -> None:
        """On a reveal generation: prequential probe → advance frontier → re-score the
        archive on the new window → free gate-failers for a retry (in that order)."""
        new_window = self._schedule[gen]
        self._prequential_score(gen, self._window.val_end_ts, new_window.val_end_ts)
        self._window = new_window
        self._rescore_archive_on_window(gen)
        released = self.controller.release_failed_fingerprints()
        if released:
            log.info("generation %d: released %d gate-failing fingerprint(s) for retry",
                     gen, released)

    def _terminal_holdout_rescore(self) -> None:
        """--final-holdout: reveal the held-out last block AFTER the final generation.

        Prequential-score it (honest — no selection ever queried it), advance the
        frontier to the full dev window and re-score the archive there — and then
        STOP: no child is proposed on this window, so the final front is ranked on
        data untouched by selection pressure.  Idempotent on resume (the prequential
        row dedups by generation; the rescore is a pure recompute, no N_trials)."""
        new_window = self._schedule[-1]
        if self._window is not None and \
                self._window.val_end_ts == new_window.val_end_ts:
            return   # resumed after the terminal step already ran
        gen_t = self.cfg.generations + 1
        log.info("final holdout: terminal rescore-only reveal %s → %s",
                 self._window.val_end_ts, new_window.val_end_ts)
        self._prequential_score(gen_t, self._window.val_end_ts,
                                new_window.val_end_ts)
        self._window = new_window
        self._rescore_archive_on_window(gen_t)
        self._checkpoint()

    def _admit(self, program: FactorProgram, *, generation: int, island: int,
               operator: str, parent_ids: Sequence[str]) -> EvaluatedGenome | None:
        """SINGLE mode: wrap one program in a genome and admit it."""
        group_id, deme_id = self.controller.coordinates(island)
        program.mechanism_group_id = group_id
        genome = Genome(
            genome_id=f"g{generation}-{program.factor_id}-{uuid.uuid4().hex[:6]}",
            programs=[program], unit="single", generation=generation,
            island=island, mechanism_group_id=group_id, deme_id=deme_id,
            parent_ids=list(parent_ids), operator=operator,
            metadata={"mechanism_group_id": group_id,
                      "mechanism": program.mechanism},
        )
        return self._admit_genome(genome)

    def _admit_set(self, programs: Sequence[FactorProgram], *, generation: int,
                   island: int, operator: str,
                   parent_ids: Sequence[str]) -> EvaluatedGenome | None:
        """SET mode: wrap a member list in a genome and admit it."""
        group_id, deme_id = self.controller.coordinates(island)
        for program in programs:
            program.mechanism_group_id = group_id
        genome = Genome(
            genome_id=f"g{generation}-set{len(programs)}-{uuid.uuid4().hex[:6]}",
            programs=list(programs), unit="set", generation=generation,
            island=island, mechanism_group_id=group_id, deme_id=deme_id,
            parent_ids=list(parent_ids), operator=operator,
            metadata={"mechanism_group_id": group_id},
        )
        return self._admit_genome(genome)

    def _admit_genome(self, genome: Genome) -> EvaluatedGenome | None:
        """Dedup → evaluate (by unit) → insert → render the reflection brief."""
        label = ",".join(genome.factor_ids)
        if self.controller.is_duplicate(genome):
            log.info("[%s] duplicate genome — skipped (not billed)", label)
            return None
        if genome.unit == "set":
            fitness = self.evaluate_set(genome.programs, genome.genome_id)
        else:
            fitness = self.evaluate_program(genome.program)
        if fitness is None:
            return None
        eg = EvaluatedGenome(genome=genome, fitness=fitness)
        self.controller.insert(eg)
        for p in genome.programs:
            self.known_ids.add(p.factor_id)
            self._program_pool.setdefault(p.factor_id, p)
        # WS5: tally this scored candidate against its mechanism — survivors AND
        # failures (the negative evidence that makes exhaustion detectable).
        if self.memory is not None:
            from quant_fund_agent.knowledge import experience
            topic = self._memory_topic(genome)
            if topic:
                genome.metadata["mechanism"] = topic
                self._mech_by_genome[genome.genome_id] = topic
            experience.record_attempt(
                self.memory, topic, survived=eg.selectable,
                marginal=eg.fitness.objective.marginal_value,
                generation=genome.generation)
        self.briefs[genome.genome_id] = self._with_memory(
            mutation_brief(
                fitness, book_size=self._book_size(),
                fixed_prediction_horizon=(
                    self.cfg.target_horizon if self.cfg.force_prediction_horizon
                    else None
                ),
            ))
        return eg

    def _memory_topic(self, genome: Genome) -> str | None:
        """Mechanism-like key for the memory: an explicit mechanism tag (GraphRAG)
        if present, else inherited from a parent, else the factor's category."""
        m = genome.metadata.get("mechanism")
        if m:
            return m
        for pid in genome.parent_ids:
            pm = self._mech_by_genome.get(pid)
            if pm:
                return pm
        for p in genome.programs:
            if p.category:
                return p.category
        return None

    def _with_memory(self, brief: str) -> str:
        """Splice the experience-memory summary into a teacher brief (WS5)."""
        return f"{brief}\n\n{self._memory_brief}" if self._memory_brief else brief

    def _write_memory(self) -> None:
        """Writeback: stamp survivors' realized performance + refresh mechanism
        coverage, then persist the per-config memory graph (WS5)."""
        if self.memory is None or self._memory_path is None:
            return
        from quant_fund_agent.knowledge import experience
        from quant_fund_agent.knowledge.empirical_edges import refresh_mechanism_coverage

        factor_ics: dict[str, float | None] = {}
        for eg in self.controller.accepted_book():
            topic = self._memory_topic(eg.genome)
            for p in eg.genome.programs:
                experience.record_survivor(
                    self.memory, p.factor_id,
                    objective=eg.fitness.objective.to_dict(),
                    val_ic=eg.fitness.diagnostics.get("val_ic"),
                    marginal_value=eg.fitness.objective.marginal_value,
                    generation=eg.genome.generation, verdict="kept",
                    mechanism_name=topic)
                factor_ics[p.factor_id] = eg.fitness.diagnostics.get("val_ic")
        refresh_mechanism_coverage(self.memory, factor_ics)
        self.memory.save(self._memory_path)
        log.info("experience memory saved → %s (%s)", self._memory_path,
                 self.memory.summary())

    # ── operators ──

    def _brief_for(self, eg: EvaluatedGenome) -> str:
        return self.briefs.get(eg.genome.genome_id) or self._with_memory(
            mutation_brief(
                eg.fitness, book_size=self._book_size(),
                fixed_prediction_horizon=(
                    self.cfg.target_horizon if self.cfg.force_prediction_horizon
                    else None
                ),
            ))

    def _book_size(self) -> int:
        archive_ids = {fid for fid, _ in self.controller.archive_programs()}
        fixed_ids = {b["factor_id"] for b in self.fixed_book}
        return len(archive_ids | fixed_ids)

    def _book_entries(self) -> list[dict[str, Any]]:
        """The accepted book as hypothesis summaries (the debate skeptic's view)."""
        fixed = [{
            "factor_id": b["factor_id"],
            "name": b.get("name", b["factor_id"]),
            "category": b.get("category", "other"),
            "trading_idea": b.get("trading_idea", ""),
            "description": b.get("description", ""),
            "prediction_horizon": b.get("prediction_horizon", self.cfg.target_horizon),
            "suggested_horizons": b.get("suggested_horizons", []),
            "expected_sign": b.get("expected_sign"),
            "source_paper_ids": b.get("source_paper_ids", []),
        } for b in self.fixed_book]
        archive = [_idea_payload(p) for eg in self.controller.archive
                   for p in eg.genome.programs]
        return fixed + archive

    def _child_llm_semantic(self, llm: Any, island: int,
                            creative: bool = False,
                            parents: list[EvaluatedGenome] | None = None,
                            ) -> tuple[FactorProgram, list[str]] | None:
        if parents is None:   # parallel path pre-selects in the main thread
            parents = self.controller.select_parents(1, island)
        if not parents:
            return None
        parent = parents[0]
        group_id, _ = self.controller.coordinates(island)
        context = self._group_context(group_id)
        if self.cfg.debate == "on":
            child = self._child_hypothesis_debate_codegen(
                llm, parent, context, creative=creative)
        else:
            prompt = build_mutation_prompt(
                parent.genome.program, self._brief_for(parent),
                context, sorted(self.known_ids), creative=creative)
            with set_llm_role("mutation"):
                child = self._parse_and_validate_child(llm, prompt, [parent])
        return (child, [parent.genome.genome_id]) if child is not None else None

    def _child_hypothesis_debate_codegen(self, hyp_llm: Any,
                                         parent: EvaluatedGenome,
                                         data_context: str,
                                         creative: bool = False,
                                         ) -> FactorProgram | None:
        """P3 agent split: Hypothesis → Debate → Codegen (debate pre-codegen,
        so tokens are never spent implementing an idea the skeptic kills)."""
        from quant_fund_agent.agents.factor_research.debate import run_debate
        from quant_fund_agent.agents.factor_research.evolution.mutation import (
            build_hypothesis_prompt,
            parse_hypothesis_response,
        )

        prompt = build_hypothesis_prompt(
            parent.genome.program, self._brief_for(parent),
            data_context, sorted(self.known_ids), creative=creative)
        try:
            with set_llm_role("hypothesis"):
                raw = parse_hypothesis_response(_invoke(hyp_llm, prompt))
        except Exception as e:  # noqa: BLE001
            log.info("hypothesis response unparsable: %s", e)
            return None

        with set_llm_role("debate"):
            verdict, final, _ = run_debate(
                self._role_llm("debate", 0.3), raw,
                data_context=data_context, book=self._book_entries())
        if verdict == "reject":
            log.info("[%s] child hypothesis rejected in debate", raw.get("factor_id"))
            return None

        idea = coerce_idea(
            final, self.cfg.target_horizon,
            force_horizon=self.cfg.force_prediction_horizon)
        if idea is None:
            return None
        idea.factor_id = self._unique_id(idea.factor_id)
        fixed_horizon = (
            self.cfg.target_horizon if self.cfg.force_prediction_horizon else None)
        with set_llm_role("codegen"):
            prog = _codegen_program(
                self._role_llm("codegen", 0.2), idea, data_context,
                expected_prediction_horizon=fixed_horizon)
        if prog is not None and not prog.source_paper_ids:
            prog.source_paper_ids = list(parent.genome.program.source_paper_ids)
        return prog

    def _child_crossover(self, llm: Any, island: int,
                         parents: list[EvaluatedGenome] | None = None,
                         ) -> tuple[FactorProgram, list[str]] | None:
        if parents is None:   # parallel path pre-selects in the main thread
            parents = self.controller.select_parents(2, island)
            if len(parents) < 2:
                return None
            a, b = parents[0], parents[1]
            if a.genome.genome_id == b.genome.genome_id:
                others = [p for p in self.controller.population(island)
                          if p.genome.genome_id != a.genome.genome_id]
                if not others:
                    return None
                b = others[int(self.rng.integers(0, len(others)))]
        else:
            if len(parents) < 2:
                return None
            a, b = parents[0], parents[1]
        group_id, _ = self.controller.coordinates(island)
        context = self._group_context(group_id)
        prompt = build_crossover_prompt(
            a.genome.program, self._brief_for(a),
            b.genome.program, self._brief_for(b),
            context, sorted(self.known_ids))
        with set_llm_role("crossover"):
            child = self._parse_and_validate_child(llm, prompt, [a, b])
        if child is None:
            return None
        if self.cfg.debate == "on":
            # Post-hoc accept/reject only (the code already exists, so a
            # "revise" verdict can't be honoured without regenerating it).
            from quant_fund_agent.agents.factor_research.debate import run_debate

            with set_llm_role("debate"):
                verdict, _, _ = run_debate(
                    self._role_llm("debate", 0.3), _idea_payload(child),
                    data_context=context, book=self._book_entries(),
                    max_revisions=0)
            if verdict == "reject":
                log.info("[%s] crossover child rejected in debate", child.factor_id)
                return None
        return child, [a.genome.genome_id, b.genome.genome_id]

    def _child_cross_group(self, llm: Any, island: int,
                           parents: list[EvaluatedGenome] | None = None,
                           other_group: int | None = None,
                           ) -> tuple[FactorProgram, list[str]] | None:
        """Synthesize parents from two reserved knowledge-graph groups."""
        group_id, _ = self.controller.coordinates(island)
        if parents is not None and other_group is not None:
            # parallel path: selection (incl. fallback decisions) done upstream
            a, b = parents[0], parents[1]
        else:
            if self.cfg.n_mechanism_groups < 2:
                return self._child_crossover(llm, island)
            local = self.controller.select_parents(1, island)
            other_groups = [
                g for g in range(self.cfg.n_mechanism_groups)
                if g != group_id and self.controller.population(mechanism_group_id=g)
            ]
            if not local or not other_groups:
                return self._child_crossover(llm, island)
            other_group = int(self.rng.choice(other_groups))
            remote = self.controller.select_parents(
                1, mechanism_group_id=other_group)
            if not remote:
                return None
            a, b = local[0], remote[0]
        context = (
            self._group_context(group_id)
            + "\n\nCROSS-GROUP SYNTHESIS\nCombine the focal group with this "
              "second knowledge-graph group when the economic interaction is "
              "coherent:\n"
            + str(self.mechanism_groups[other_group].get("focus") or "")
        )
        prompt = build_crossover_prompt(
            a.genome.program, self._brief_for(a),
            b.genome.program, self._brief_for(b),
            context, sorted(self.known_ids))
        with set_llm_role("crossover"):
            child = self._parse_and_validate_child(llm, prompt, [a, b])
        if child is None:
            return None
        child.mechanism = (
            f"hybrid:{group_id}+{other_group}")
        return child, [a.genome.genome_id, b.genome.genome_id]

    def _parse_and_validate_child(self, llm: Any, prompt: str,
                                  parents: list[EvaluatedGenome]) -> FactorProgram | None:
        """LLM call → parsed, uniquified, compiled + smoke-tested child.

        One feedback retry covers BOTH failure kinds (unparsable JSON and
        validation/smoke errors) — the same self-correction loop the oneshot
        codegen uses.
        """
        from quant_fund_agent.agents.factor_research.evolution.mutation import (
            rewrite_factor_id,
        )
        from quant_fund_agent.factors.inmem import compile_factor

        current_prompt = prompt
        program: FactorProgram | None = None
        for attempt in range(2):
            try:
                program = parse_child_response(_invoke(llm, current_prompt))
                expected_horizon = (
                    self.cfg.target_horizon
                    if self.cfg.force_prediction_horizon else None
                )
                if expected_horizon is not None:
                    program.prediction_horizon = expected_horizon
                    program.suggested_horizons = [expected_horizon]
                fid = self._unique_id(program.factor_id)
                if fid != program.factor_id:
                    program.code = rewrite_factor_id(program.code,
                                                     program.factor_id, fid)
                    program.factor_id = fid
                compile_factor(
                    program.code, program.factor_id, smoke=True,
                    expected_prediction_horizon=expected_horizon)
                break
            except Exception as e:  # noqa: BLE001 — feed the error back once
                from quant_fund_agent.llm import LLMBudgetExceeded

                if isinstance(e, LLMBudgetExceeded):
                    raise   # a budget stop is not a fixable code error — never retry
                program = None
                if attempt == 0:
                    log.info("child attempt failed (%s) — retrying with feedback", e)
                    current_prompt = (
                        prompt + "\n\nYOUR PREVIOUS ATTEMPT FAILED\n"
                        "----------------------------\n"
                        f"    {e}\n\nFix exactly this issue and respond again "
                        "with the same strict JSON schema.")
                else:
                    log.info("child failed twice (%s) — dropped", e)
        if program is None:
            return None
        program.source_paper_ids = sorted({
            pid for p in parents for pid in p.genome.program.source_paper_ids})
        return program

    # ── parallel child proposal (llm_workers > 1) ─────────────────────────────
    #
    # Children within a generation are independent given generation-start
    # state, so only the LLM round-trips fan out to threads.  All RNG draws
    # (operator, creative flag, parent selection incl. fallback decisions)
    # happen SEQUENTIALLY in the main thread via _prep_child_job, and
    # admissions/evaluation happen sequentially in child order afterwards —
    # selection and scoring semantics are identical to the sequential path;
    # only wall-clock (and the LLM's own sampling) differs.

    def _prep_child_job(self, op: str, island: int) -> dict[str, Any] | None:
        """Main-thread RNG phase: returns a thread-safe job spec (or None)."""
        if op == "jitter":
            # no LLM involved — run it here so its RNG use stays sequential
            return {"kind": "done", "island": island, "op_label": op,
                    "result": self._child_jitter(island)}
        if op == "crossover":
            parents = self.controller.select_parents(2, island)
            if len(parents) < 2:
                return None
            a, b = parents[0], parents[1]
            if a.genome.genome_id == b.genome.genome_id:
                others = [p for p in self.controller.population(island)
                          if p.genome.genome_id != a.genome.genome_id]
                if not others:
                    return None
                b = others[int(self.rng.integers(0, len(others)))]
            return {"kind": "crossover", "island": island, "op_label": op,
                    "parents": [a, b]}
        if op == "cross_group":
            if self.cfg.n_mechanism_groups < 2:
                job = self._prep_child_job("crossover", island)
                if job is not None:
                    job["op_label"] = op
                return job
            group_id, _ = self.controller.coordinates(island)
            local = self.controller.select_parents(1, island)
            other_groups = [
                g for g in range(self.cfg.n_mechanism_groups)
                if g != group_id
                and self.controller.population(mechanism_group_id=g)
            ]
            if not local or not other_groups:
                job = self._prep_child_job("crossover", island)
                if job is not None:
                    job["op_label"] = op
                return job
            other_group = int(self.rng.choice(other_groups))
            remote = self.controller.select_parents(
                1, mechanism_group_id=other_group)
            if not remote:
                return None
            return {"kind": "cross_group", "island": island, "op_label": op,
                    "parents": [local[0], remote[0]],
                    "other_group": other_group}
        # semantic mutation (default op)
        creative = False
        if self.cfg.creative_frac > 0:
            creative = bool(float(self.rng.random()) < self.cfg.creative_frac)
        parents = self.controller.select_parents(1, island)
        if not parents:
            return None
        return {"kind": "semantic", "island": island,
                "op_label": "llm_semantic_creative" if creative else op,
                "parents": parents, "creative": creative}

    def _run_child_job(self, job: dict[str, Any]
                       ) -> tuple[FactorProgram, list[str]] | None:
        """Worker phase: LLM round-trips only; no shared-state mutation."""
        from quant_fund_agent.llm import budget_exhausted

        kind = job["kind"]
        if kind == "done":
            return job.get("result")
        if budget_exhausted():
            # hard block: queued proposals stop spending the moment the
            # ceiling trips — only calls already in flight can overshoot
            return None
        try:
            llm = self._role_llm("hypothesis", 0.6)
            if kind == "refine":
                return self._child_refine(llm, job["island"],
                                          job["parents"][0])
            if kind == "crossover":
                return self._child_crossover(llm, job["island"],
                                             parents=job["parents"])
            if kind == "cross_group":
                return self._child_cross_group(llm, job["island"],
                                               parents=job["parents"],
                                               other_group=job["other_group"])
            return self._child_llm_semantic(llm, job["island"],
                                            creative=job["creative"],
                                            parents=job["parents"])
        except Exception as e:  # noqa: BLE001 — one child must not kill the gen
            from quant_fund_agent.llm import LLMBudgetExceeded

            if isinstance(e, LLMBudgetExceeded):
                raise
            log.warning("parallel child proposal failed (%s)", e)
            return None

    def _child_jitter(self, island: int) -> tuple[FactorProgram, list[str]] | None:
        parents = self.controller.select_parents(1, island)
        if not parents:
            return None
        parent = parents[0]
        new_id = self._unique_id(f"{parent.genome.program.factor_id}_j")
        child = random_jitter_child(parent.genome.program, self.rng, new_id)
        return (child, [parent.genome.genome_id]) if child is not None else None

    # ── refinement variant (--variant refine): non-evolutionary ablation ──

    def _register_lineage(self, island: int, eg: EvaluatedGenome,
                          refines: int = 0, last_gen: int = -1) -> None:
        self._refine_entries.setdefault(island, []).append(
            {"eg": eg, "refines": int(refines), "last_gen": int(last_gen)})

    def _next_refine_entry(self, island: int, gen: int) -> dict[str, Any] | None:
        """The pending lineage to refine on this slot: fewest refinements first
        (FIFO among ties), at most one refinement per lineage per generation."""
        entries = self._refine_entries.get(island, [])
        eligible = [e for e in entries
                    if e["refines"] < max(0, self.cfg.refine_rounds)
                    and e["last_gen"] != gen]
        if not eligible:
            return None
        return min(eligible, key=lambda e: e["refines"])

    def _prune_refine_entries(self) -> None:
        for island, entries in self._refine_entries.items():
            self._refine_entries[island] = [
                e for e in entries
                if e["refines"] < max(0, self.cfg.refine_rounds)]

    def _child_refine(self, llm: Any, island: int, parent: EvaluatedGenome,
                      ) -> tuple[FactorProgram, list[str]] | None:
        """Refine the SAME factor against its latest evaluation report."""
        group_id, _ = self.controller.coordinates(island)
        prompt = build_refine_prompt(
            parent.genome.program, self._brief_for(parent),
            self._group_context(group_id), sorted(self.known_ids))
        with set_llm_role("mutation"):
            child = self._parse_and_validate_child(llm, prompt, [parent])
        return (child, [parent.genome.genome_id]) if child is not None else None

    def _prep_refine_slot(self, island: int, gen: int) -> dict[str, Any] | None:
        """Main-thread RNG/selection phase for one refine-variant child slot.

        Returns a job for the worker pool, ``{"kind": "fresh_seed"}`` when the
        deme has no refinement work left (handled per-group afterwards), or
        ``None`` when nothing can be proposed.
        """
        # Occasional explicit cross-group synthesis — the ONLY combination
        # operator in this variant.  No fallback to same-group crossover: when
        # cross-group parents are unavailable the slot refines instead.
        if self.cfg.n_mechanism_groups > 1 and self.cfg.p_cross_group > 0 \
                and float(self.rng.random()) < self.cfg.p_cross_group:
            group_id, _ = self.controller.coordinates(island)
            local = self.controller.select_parents(1, island)
            other_groups = [
                g for g in range(self.cfg.n_mechanism_groups)
                if g != group_id
                and self.controller.population(mechanism_group_id=g)
            ]
            if local and other_groups:
                other_group = int(self.rng.choice(other_groups))
                remote = self.controller.select_parents(
                    1, mechanism_group_id=other_group)
                if remote:
                    return {"kind": "cross_group", "island": island,
                            "op_label": "cross_group",
                            "parents": [local[0], remote[0]],
                            "other_group": other_group}
        entry = self._next_refine_entry(island, gen)
        if entry is None:
            return {"kind": "fresh_seed", "island": island}
        # Bill the attempt now (a failed child still consumes a round) and
        # refresh the brief from the CURRENT fitness so refinement reacts to
        # the latest progressive-reveal rescore, not the stale first report.
        entry["refines"] += 1
        entry["last_gen"] = gen
        parent = entry["eg"]
        self.briefs[parent.genome.genome_id] = self._with_memory(
            mutation_brief(
                parent.fitness, book_size=self._book_size(),
                fixed_prediction_horizon=(
                    self.cfg.target_horizon
                    if self.cfg.force_prediction_horizon else None),
            ))
        return {"kind": "refine", "island": island, "op_label": "refine",
                "parents": [parent], "entry": entry}

    def _seed_fresh_lineages(self, gen: int, needed_by_group: dict[int, int],
                             ) -> int:
        """Mid-run graphrag re-seeding: one brainstorm/codegen batch per group
        that ran out of refinement work, each survivor starting a new lineage."""
        made = 0
        for group_id in sorted(needed_by_group):
            n = needed_by_group[group_id]
            if n <= 0:
                continue
            spec = self.mechanism_groups[group_id % len(self.mechanism_groups)]
            per_cfg = replace(self.cfg, n_seed_ideas=n)
            try:
                programs = seed_programs(
                    per_cfg, self._group_context(group_id), self.known_ids,
                    self.fields, mechanism_group=spec)
            except Exception as e:  # noqa: BLE001 — a bad seeding batch must
                from quant_fund_agent.llm import LLMBudgetExceeded
                if isinstance(e, LLMBudgetExceeded):
                    raise
                log.warning("fresh seeding failed for group %d (%s)", group_id, e)
                continue
            self.known_ids.update(p.factor_id for p in programs)
            for k, prog in enumerate(programs):
                deme = k % self.cfg.effective_demes_per_group
                island = self.controller.flat_island(group_id, deme)
                eg = self._admit(prog, generation=gen, island=island,
                                 operator="seed", parent_ids=[])
                if eg is not None:
                    made += 1
                    self._register_lineage(island, eg)
        return made

    def _refine_generation(self, gen: int) -> tuple[int, int]:
        """One refine-variant generation → (children admitted, slots offered).

        Slot mix per deme (``children_per_deme`` slots): occasional cross-group
        synthesis, else one refinement of the deme's least-refined pending
        lineage, else a fresh graphrag seed for the deme's mechanism group.
        LLM round-trips fan out to ``llm_workers`` threads exactly like the
        evolve path (RNG/selection sequential, admissions sequential).
        """
        cfg = self.cfg
        made = 0
        slots = [island
                 for island in range(len(self.controller.islands))
                 for _ in range(max(1, cfg.children_per_deme
                                    if cfg.children_per_deme is not None else 1))]
        jobs: list[dict[str, Any]] = []
        fresh_needed: dict[int, int] = {}
        for island in slots:
            job = self._prep_refine_slot(island, gen)
            if job is None:
                continue
            if job["kind"] == "fresh_seed":
                group_id, _ = self.controller.coordinates(island)
                fresh_needed[group_id] = fresh_needed.get(group_id, 0) + 1
                continue
            jobs.append(job)
        results = _parallel_map(self._run_child_job, jobs, cfg.llm_workers)
        for job, result in zip(jobs, results):
            if result is None:
                continue
            child, parent_ids = result
            eg = self._admit(child, generation=gen, island=job["island"],
                             operator=job["op_label"], parent_ids=parent_ids)
            if eg is None:
                continue
            made += 1
            if job["kind"] == "refine":
                # The lineage continues from its latest version.
                job["entry"]["eg"] = eg
            else:
                # A cross-group synthesis starts a lineage of its own.
                self._register_lineage(job["island"], eg)
        self._prune_refine_entries()
        self._pending_ids.clear()
        made += self._seed_fresh_lineages(gen, fresh_needed)
        return made, len(slots)

    # ── refine-variant persistence (resume safety) ──

    def _save_refine_state(self) -> None:
        payload = {
            str(island): [{"genome_id": e["eg"].genome.genome_id,
                           "refines": e["refines"],
                           "last_gen": e["last_gen"]}
                          for e in entries]
            for island, entries in self._refine_entries.items()
        }
        (self.out_dir / "refine_state.json").write_text(
            json.dumps(payload, indent=2))

    def _load_refine_state(self) -> None:
        path = self.out_dir / "refine_state.json"
        self._refine_entries = {}
        if not path.exists():
            log.warning("refine variant resume: %s missing — every deme "
                        "restarts with fresh seeding", path)
            return
        by_id: dict[str, EvaluatedGenome] = {}
        for eg in [*self.controller.population(), *self.controller.archive,
                   *self.controller.kept_pool]:
            by_id.setdefault(eg.genome.genome_id, eg)
        payload = json.loads(path.read_text())
        dropped = 0
        for island_key, entries in payload.items():
            island = int(island_key)
            for row in entries:
                eg = by_id.get(str(row.get("genome_id")))
                if eg is None:
                    dropped += 1
                    continue
                self._register_lineage(island, eg,
                                       refines=int(row.get("refines", 0)),
                                       last_gen=int(row.get("last_gen", -1)))
        n = sum(len(v) for v in self._refine_entries.values())
        log.info("refine variant resume: restored %d pending lineage(s)%s",
                 n, f" ({dropped} dropped — genome pruned everywhere)"
                 if dropped else "")

    # ── SET-mode operators (structural; deterministic given the seed) ──

    def _child_set(self, op: str, island: int,
                   ) -> tuple[list[FactorProgram], list[str]] | None:
        """One SET child via ``structural`` (add/drop/replace member),
        ``splice`` (union-sample two local parents), ``cross_group_splice``,
        or ``member_jitter``.

        All three are programmatic — the LLM's creativity enters SET mode
        through the generation-0 members (and any future member-level LLM
        mutation, a documented extension); the set-level search explores
        *composition* space, which is what the unit is for.
        """
        if op == "cross_group_splice":
            group_id, _ = self.controller.coordinates(island)
            local = self.controller.select_parents(1, island)
            other_groups = [
                g for g in range(self.cfg.n_mechanism_groups)
                if g != group_id and self.controller.population(mechanism_group_id=g)
            ]
            if not local or not other_groups:
                return None
            other_group = int(self.rng.choice(other_groups))
            remote = self.controller.select_parents(
                1, mechanism_group_id=other_group)
            if not remote:
                return None
            a, b = local[0], remote[0]
            pool = {p.factor_id: p
                    for p in [*a.genome.programs, *b.genome.programs]}
            k = max(1, min(len(pool),
                           max(len(a.genome.programs), len(b.genome.programs))))
            chosen = self.rng.choice(sorted(pool), size=k, replace=False)
            # Copy members before the focal-group admission stamps provenance;
            # never mutate the remote parent programs in place.
            return ([FactorProgram.from_dict(pool[str(fid)].to_dict())
                     for fid in chosen],
                    [a.genome.genome_id, b.genome.genome_id])

        parents = self.controller.select_parents(2 if op == "splice" else 1, island)
        if not parents:
            return None
        a = parents[0]
        members = list(a.genome.programs)

        if op == "splice":
            b = parents[1] if len(parents) > 1 else None
            if b is None or b.genome.genome_id == a.genome.genome_id:
                others = [p for p in self.controller.population(island)
                          if p.genome.genome_id != a.genome.genome_id]
                if not others:
                    return None
                b = others[int(self.rng.integers(0, len(others)))]
            pool = {p.factor_id: p
                    for p in [*a.genome.programs, *b.genome.programs]}
            k = max(1, min(len(pool),
                           max(len(a.genome.programs), len(b.genome.programs))))
            chosen = self.rng.choice(sorted(pool), size=k, replace=False)
            return ([pool[str(fid)] for fid in chosen],
                    [a.genome.genome_id, b.genome.genome_id])

        if op == "member_jitter":
            for idx in self.rng.permutation(len(members)):
                new_id = self._unique_id(f"{members[int(idx)].factor_id}_j")
                child = random_jitter_child(members[int(idx)], self.rng, new_id)
                if child is not None:
                    members[int(idx)] = child
                    return members, [a.genome.genome_id]
            return None

        # structural: add / drop / replace within the same mechanism group.
        # Cross-group composition is deliberately reserved for the explicit
        # cross-group synthesis operator rather than leaking through this pool.
        group_id, _ = self.controller.coordinates(island)
        current = set(a.genome.factor_ids)
        external = [p for fid, p in sorted(self._program_pool.items())
                    if fid not in current and p.mechanism_group_id == group_id]
        moves = []
        if external:
            moves.extend(["add", "replace"])
        if len(members) > 1:
            moves.append("drop")
        if not moves:
            return None
        move = str(self.rng.choice(sorted(set(moves))))
        if move in ("drop", "replace"):
            members.pop(int(self.rng.integers(0, len(members))))
        if move in ("add", "replace"):
            members.append(external[int(self.rng.integers(0, len(external)))])
        return members, [a.genome.genome_id]

    # ── checkpointing ──

    def _checkpoint(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.controller.save(self.out_dir / "state.json")
        (self.out_dir / "run_config.json").write_text(
            json.dumps(self.cfg.to_dict(), indent=2))
        try:
            (self.out_dir / "llm_usage.json").write_text(
                json.dumps(usage_summary(), indent=2, default=str))
        except Exception as e:  # noqa: BLE001 — metering must never kill a run
            log.warning("could not write llm_usage.json: %s", e)
        (self.out_dir / "mechanism_groups.json").write_text(
            json.dumps(self.mechanism_groups, indent=2, default=str))
        with (self.out_dir / "lineage.jsonl").open("w") as fh:
            for row in self.controller.lineage:
                fh.write(json.dumps(row, default=str) + "\n")
        if self.cfg.progressive and self._window is not None:
            # The reveal frontier the recorded generation implies — a cheap resume
            # corruption guard (the schedule is otherwise a pure function of config).
            (self.out_dir / "progressive.json").write_text(json.dumps({
                "generation": self.controller.generation,
                "frontier_ts": self._window.val_end_ts,
                "visible_end": self._window.visible_end,
            }, indent=2))
        if self.cfg.variant == "refine":
            self._save_refine_state()

    def _write_gen_quality(self) -> None:
        """Append one cheap per-generation quality row to ``gen_quality.jsonl``.

        Computed purely from the controller's in-memory archive (no model fits).
        Resume-safe: one row per generation, deduped against the existing file.
        """
        gen = int(self.controller.generation)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self._gen_quality_done is None:
            self._gen_quality_done = self._jsonl_generations(
                self.out_dir / "gen_quality.jsonl")
        if gen in self._gen_quality_done:
            return
        archive = list(self.controller.archive)

        def _axis(name: str) -> list[float]:
            vals: list[float] = []
            for eg in archive:
                v = getattr(eg.fitness.objective, name, None)
                if v is not None:
                    vals.append(float(v))
            return vals

        marginal = _axis("marginal_value")
        novelty = _axis("structural_novelty")
        row = {
            "generation": gen,
            "archive_size_total": len(archive),
            "archive_size_per_group": [
                len(group)
                for group in getattr(self.controller, "group_archives", [])
            ],
            "kept_pool_size": len(getattr(self.controller, "kept_pool", [])),
            "n_trials": self.controller.n_trials,
            "mean_marginal_value": (
                sum(marginal) / len(marginal) if marginal else None),
            "max_marginal_value": max(marginal) if marginal else None,
            "mean_structural_novelty": (
                sum(novelty) / len(novelty) if novelty else None),
        }
        with (self.out_dir / "gen_quality.jsonl").open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        self._gen_quality_done.add(gen)

    # ── main drive ──

    def _reload_lineage(self) -> None:
        """Reload lineage.jsonl on resume (the controller checkpoint does not
        carry it) so the "w"-rewrite at the next checkpoint keeps the full
        history instead of truncating it to the resumed rows."""
        lineage_path = self.out_dir / "lineage.jsonl"
        if self.controller.lineage or not lineage_path.exists():
            return
        rows: list[dict[str, Any]] = []
        for line in lineage_path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("skipping corrupt lineage row on resume: %r",
                            line[:120])
        self.controller.lineage = rows
        log.info("resume: reloaded %d lineage row(s) from %s",
                 len(rows), lineage_path)

    def _admit_seed_group(self, group_id: int,
                          group_programs: Sequence[FactorProgram]) -> None:
        """Admit one mechanism group's generation-0 seeds (single or SET)."""
        cfg = self.cfg
        if cfg.unit == "set":
            size = max(1, cfg.set_size)
            for k in range(0, len(group_programs), size):
                chunk = list(group_programs[k:k + size])
                if chunk:
                    deme = (k // size) % cfg.effective_demes_per_group
                    island = self.controller.flat_island(group_id, deme)
                    self._admit_set(chunk, generation=0, island=island,
                                    operator="seed", parent_ids=[])
        else:
            for k, prog in enumerate(group_programs):
                deme = k % cfg.effective_demes_per_group
                island = self.controller.flat_island(group_id, deme)
                eg = self._admit(prog, generation=0, island=island,
                                 operator="seed", parent_ids=[])
                if cfg.variant == "refine" and eg is not None:
                    self._register_lineage(island, eg)

    def run(self, initial_programs: Sequence[FactorProgram] | None = None,
            *, resume: bool = False) -> dict[str, Any]:
        """Run the whole evolutionary search; returns a summary dict.

        ``initial_programs`` (mostly for tests / resumed runs) bypasses the LLM
        seed brainstorm.  ``resume`` continues an already-seeded controller (loaded
        from a checkpoint) from ``controller.generation + 1`` instead of re-seeding.
        """
        cfg = self.cfg
        t0 = time.time()
        self._load_known_ids()

        # A replaced/reloaded controller (resume) must feed the same event sink.
        if hasattr(self.controller, "event_sink") \
                and getattr(self.controller, "event_sink", None) is None:
            self.controller.event_sink = self._lineage_event

        if cfg.progressive:
            # Build the reveal schedule and set the window in force (generation 0 on a
            # fresh run, or the resumed generation).  Corruption guard: a resumed
            # controller must land on the frontier its recorded generation implies.
            self._init_progressive()

        resuming = (resume and self.controller.generation > 0
                    and bool(self.controller.population()))
        if resuming:
            start_gen = self.controller.generation
            # The controller checkpoint does not carry the lineage; reload it from
            # lineage.jsonl so the "w"-rewrite at the next checkpoint keeps the
            # full history instead of truncating it to the resumed rows.
            self._reload_lineage()
            log.info("resuming from generation %d: population=%d, archive=%d, "
                     "n_trials=%d", start_gen, len(self.controller.population()),
                     len(self.controller.archive), self.controller.n_trials)
            if cfg.variant == "refine":
                self._load_refine_state()
        else:
            start_gen = 0
            # ── generation 0: seed ──
            # Mid-seeding resume: a run killed DURING generation-0 seeding left a
            # checkpoint with generation == 0 and a partial population (seeding
            # checkpoints after every mechanism group below).  Restore run state
            # and skip the groups that were already seeded, so a restart never
            # re-spends their LLM calls.
            mid_seed_resume = resume and bool(self.controller.population())
            if mid_seed_resume:
                self._reload_lineage()
                if cfg.variant == "refine":
                    self._load_refine_state()
                for eg in self.controller.population():
                    for p in eg.genome.programs:
                        self.known_ids.add(p.factor_id)
                        self._program_pool.setdefault(p.factor_id, p)
                log.info("resuming mid-seeding: %d genome(s), %d trial(s) "
                         "already admitted", len(self.controller.population()),
                         self.controller.n_trials)
            grouped_programs: dict[int, list[FactorProgram]] = {}
            if initial_programs is not None:
                # Tests/manual resumes can supply programs directly.  Respect an
                # explicit program group tag; otherwise distribute by group.
                for k, program in enumerate(initial_programs):
                    tagged = bool(program.mechanism) or program.mechanism_group_id != 0
                    group_id = (program.mechanism_group_id if tagged else k) \
                        % max(1, cfg.n_mechanism_groups)
                    program.mechanism_group_id = group_id
                    grouped_programs.setdefault(group_id, []).append(program)
                for prog in [p for g in sorted(grouped_programs)
                             for p in grouped_programs[g]]:
                    self._program_pool.setdefault(prog.factor_id, prog)
                for group_id, group_programs in grouped_programs.items():
                    self._admit_seed_group(group_id, group_programs)
            else:
                self.mechanism_groups = (self._resolved_groups
                                         or resolve_mechanism_groups(cfg, self.fields))
                per_group_cfg = replace(
                    cfg, n_seed_ideas=cfg.effective_seed_ideas_per_group)
                for spec in self.mechanism_groups:
                    group_id = int(spec["mechanism_group_id"])
                    if mid_seed_resume and self.controller.population(
                            mechanism_group_id=group_id):
                        log.info("group %d already seeded before restart — "
                                 "skipped", group_id)
                        continue
                    programs_for_group = seed_programs(
                        per_group_cfg, self._group_context(group_id),
                        self.known_ids, self.fields, mechanism_group=spec)
                    self.known_ids.update(
                        p.factor_id for p in programs_for_group)
                    for prog in programs_for_group:
                        self._program_pool.setdefault(prog.factor_id, prog)
                    self._admit_seed_group(group_id, programs_for_group)
                    # Kill-resilience: a restart re-seeds at most one group.
                    self._checkpoint()
            self._write_gen_quality()
            self._checkpoint()
            log.info("generation 0: population=%d, archive=%d, n_trials=%d",
                     len(self.controller.population()), len(self.controller.archive),
                     self.controller.n_trials)
            if not self.controller.population():
                log.warning("empty seed population — aborting run")
                return self.summary(time.time() - t0)

        if cfg.unit == "set":
            ops: list[tuple[str, float]] = [
                ("structural", max(0.0, cfg.p_llm_semantic)),
                ("splice", max(0.0, cfg.p_crossover)),
                ("cross_group_splice", max(0.0, cfg.p_cross_group)
                 if cfg.n_mechanism_groups > 1 else 0.0),
                ("member_jitter", max(0.0, cfg.p_jitter)),
            ]
        else:
            ops = [
                ("llm_semantic", max(0.0, cfg.p_llm_semantic)),
                ("crossover", max(0.0, cfg.p_crossover)),
                ("cross_group", max(0.0, cfg.p_cross_group)
                 if cfg.n_mechanism_groups > 1 else 0.0),
                ("jitter", max(0.0, cfg.p_jitter)),
            ]
        total_p = sum(p for _, p in ops) or 1.0
        op_names = [n for n, _ in ops]
        op_probs = [p / total_p for _, p in ops]

        if cfg.llm_workers > 1 and cfg.unit != "set":
            # Pre-warm the per-role chat-model cache so worker threads only
            # ever READ it (lazy construction from multiple threads races).
            for role, temp in (("hypothesis", 0.6), ("debate", 0.3),
                               ("codegen", 0.2)):
                try:
                    self._role_llm(role, temp)
                except Exception as e:  # noqa: BLE001 — surfaced on first use
                    log.warning("could not pre-build %s LLM (%s)", role, e)

        for gen in range(start_gen + 1, cfg.generations + 1):
            self.controller.generation = gen
            # progressive reveal: on a reveal generation, run the prequential probe,
            # advance the frontier, re-score the archive and free gate-failers for a
            # retry — all BEFORE any child is proposed this generation.
            if cfg.progressive and self._schedule[gen].reveal:
                self._advance_reveal(gen)
            if cfg.variant == "refine":
                # Non-evolutionary ablation: refine-in-place / cross-group /
                # fresh graphrag seeding; no tournament ops, no migration.
                made, offered = self._refine_generation(gen)
                self._write_gen_quality()
                self._checkpoint()
                log.info("generation %d (refine): %d/%d children admitted; "
                         "archive=%d; n_trials=%d", gen, made, offered,
                         len(self.controller.archive), self.controller.n_trials)
                continue
            made = 0
            if cfg.children_per_deme is not None:
                child_islands = [
                    island
                    for island in range(len(self.controller.islands))
                    for _ in range(max(1, cfg.children_per_deme))
                ]
            else:
                child_islands = [
                    k % len(self.controller.islands)
                    for k in range(cfg.children_per_generation)
                ]
            if cfg.llm_workers > 1 and cfg.unit != "set":
                # Parallel proposals: RNG/selection in the main thread,
                # LLM round-trips pooled, admissions sequential in order.
                jobs = [self._prep_child_job(
                            str(self.rng.choice(op_names, p=op_probs)), island)
                        for island in child_islands]
                live = [j for j in jobs if j is not None]
                results = iter(_parallel_map(self._run_child_job, live,
                                             cfg.llm_workers))
                for job in jobs:
                    if job is None:
                        continue
                    made_child = next(results)
                    if made_child is None:
                        continue
                    child, parent_ids = made_child
                    eg = self._admit(child, generation=gen,
                                     island=job["island"],
                                     operator=job["op_label"],
                                     parent_ids=parent_ids)
                    if eg is not None:
                        made += 1
                self._pending_ids.clear()
                child_loop_done = True
            else:
                child_loop_done = False
            for island in (child_islands if not child_loop_done else []):
                op = str(self.rng.choice(op_names, p=op_probs))
                if cfg.unit == "set":
                    made_set = self._child_set(op, island)
                    if made_set is None:
                        continue
                    members, parent_ids = made_set
                    eg = self._admit_set(members, generation=gen, island=island,
                                         operator=op, parent_ids=parent_ids)
                    if eg is not None:
                        made += 1
                    continue
                # the mutating LLM is built lazily (and cached) so jitter-only
                # and SET runs never construct a chat model at all
                creative = False
                if op == "jitter":
                    made_child = self._child_jitter(island)
                elif op == "crossover":
                    made_child = self._child_crossover(
                        self._role_llm("hypothesis", 0.6), island)
                elif op == "cross_group":
                    made_child = self._child_cross_group(
                        self._role_llm("hypothesis", 0.6), island)
                else:
                    # --creative-frac: with this probability the semantic child is
                    # asked for a genuinely novel mechanism (RNG drawn only when the
                    # flag is on, so the default stream stays byte-identical).
                    if cfg.creative_frac > 0:
                        creative = bool(
                            float(self.rng.random()) < cfg.creative_frac)
                    made_child = self._child_llm_semantic(
                        self._role_llm("hypothesis", 0.6), island,
                        creative=creative)
                if made_child is None:
                    continue
                child, parent_ids = made_child
                op_label = "llm_semantic_creative" if creative else op
                eg = self._admit(child, generation=gen, island=island,
                                 operator=op_label, parent_ids=parent_ids)
                if eg is not None:
                    made += 1
            if cfg.effective_demes_per_group > 1 \
                    and gen % max(1, cfg.migration_every) == 0:
                moved = self.controller.migrate()
                log.info("generation %d: migrated %d elite(s)", gen, moved)
            self._write_gen_quality()
            self._checkpoint()
            log.info("generation %d: %d/%d children admitted; archive=%d; n_trials=%d",
                     gen, made, len(child_islands),
                     len(self.controller.archive), self.controller.n_trials)

        if cfg.progressive and cfg.final_holdout and self._schedule is not None:
            self._terminal_holdout_rescore()
        self._write_memory()   # WS5: persist survivors + attempt tallies for the next run
        return self.summary(time.time() - t0)

    def summary(self, elapsed: float) -> dict[str, Any]:
        return {
            "generations": self.controller.generation,
            "n_trials": self.controller.n_trials,
            "population": len(self.controller.population()),
            "fixed_book_size": len(self.fixed_book),
            "mechanism_groups": [
                {
                    **spec,
                    "population": len(self.controller.population(
                        mechanism_group_id=int(spec["mechanism_group_id"]))),
                    "archive_size": len(self.controller.group_archive(
                        int(spec["mechanism_group_id"]))),
                }
                for spec in self.mechanism_groups
            ],
            "archive": [
                {"factor_ids": eg.genome.factor_ids,
                 "genome_id": eg.genome.genome_id,
                 "unit": eg.genome.unit,
                 "operator": eg.genome.operator,
                 "generation": eg.genome.generation,
                 "mechanism_group_id": eg.genome.mechanism_group_id,
                 "deme_id": eg.genome.deme_id,
                 "objective": eg.fitness.objective.to_dict()}
                for eg in self.controller.archive
            ],
            "n_eval_failures": len(self.failures),
            "elapsed_sec": round(elapsed, 1),
            "llm_usage": usage_summary(),
        }


# ── persisting the survivors ──────────────────────────────────────────────────

def persist_archive(controller: EvolutionController, *, session_id: str,
                    target_horizon: int, cutoff_date: str | None = None,
                    data_dir: str = "ticker_data", n_tickers: int | None = 15,
                    curation: str = "archive", n_keep: int | None = None,
                    is_frac: float = 0.6, val_frac: float = 0.2,
                    fields: Sequence[str] | None = None,
                    marginal_model: str = "gradient_boosting",
                    selection_deflation: str = "off",
                    engine: str = "evolution",
                    model_label: str | None = None,
                    ) -> dict[str, list[str]]:
    """Materialise the final book into real factor files + DB records.

    Uses the *same* materialise / IC-backtest / persist path as the oneshot
    engine, so an evolution prerun is indistinguishable downstream (comparison
    harness, fund runs) from any other prerun.  Records carry the evolution
    provenance (generation, operator, parents, objective vector, gates) in
    ``metadata.evolution``.

    Two-stage curation (Lever 2): with ``curation == "archive"`` (the default)
    the persisted book is the Pareto **archive** — today's behaviour.  With
    ``greedy`` / ``elastic_net`` the book is instead the **curated kept-pool**
    (every gate-passing factor, then curated to a chosen set / ``n_keep`` size),
    so a good factor is no longer dropped merely for being dominated.
    """
    from quant_fund_agent.llm import resolve_research_model, resolve_research_provider
    from quant_fund_agent.mcp import research_client
    from quant_fund_agent.schemas import (
        BacktestMetrics,
        FactorRecord,
        FactorSource,
        FactorStatus,
        TradingIdeaCategory,
    )

    # ── choose source: reserved group archives or the full kept-pool ──
    # ``accepted_book()`` is the union of mechanism-group Pareto archives.
    use_pool = curation != "archive" and bool(controller.kept_pool)
    source = controller.kept_pool if use_pool else controller.accepted_book()
    prog_by_fid: dict[str, tuple[EvaluatedGenome, FactorProgram]] = {}
    for eg in source:
        for prog in eg.genome.programs:
            prog_by_fid.setdefault(prog.factor_id, (eg, prog))

    import os

    # Fail-closed by default: a curation/publish failure (or an empty curated /
    # deflated book) aborts the persist so a broken stage can't silently ship
    # the un-curated pool.  ``QF_PERSIST_FAIL_OPEN=1`` restores the historical
    # fallback, flagged in each persisted factor's ``metadata.evolution``.
    fail_open = os.getenv("QF_PERSIST_FAIL_OPEN") == "1"
    curation_failed = False
    publish_failed = False

    kept_ids = list(prog_by_fid)
    curation_info: dict[str, Any] | None = None
    if use_pool and prog_by_fid:
        pool = [{"factor_id": fid, "code": prog.code}
                for fid, (_, prog) in prog_by_fid.items()]
        try:
            res = research_client.curate_book(
                pool, mode=curation, n_keep=n_keep, target_horizon=target_horizon,
                is_frac=is_frac, val_frac=val_frac, cutoff_date=cutoff_date,
                data_dir=data_dir, n_tickers=n_tickers,
                fields=list(fields) if fields else None,
                marginal_model=marginal_model)
        except Exception as e:  # noqa: BLE001 — surfaced as a stage failure below
            res = {"ok": False, "error": repr(e)}
        curated = ([fid for fid in res.get("kept_factor_ids", [])
                    if fid in prog_by_fid] if res.get("ok") else [])
        if curated:
            kept_ids = curated
            curation_info = {k: res[k] for k in
                             ("mode", "combined_ic", "selection_frequency", "n_pool")
                             if k in res}
            log.info("curation (%s) kept %d/%d factors from the pool",
                     curation, len(kept_ids), len(prog_by_fid))
        else:
            err = (res.get("error") if not res.get("ok")
                   else "curation returned an empty book")
            if not fail_open:
                raise RuntimeError(
                    f"persist_archive: curation stage ({curation}) failed: {err}. "
                    "Set QF_PERSIST_FAIL_OPEN=1 to persist the full pool instead.")
            curation_failed = True
            log.warning(
                "!!! curation (%s) failed: %s — QF_PERSIST_FAIL_OPEN=1, "
                "persisting the FULL UN-CURATED pool (flagged in metadata)",
                curation, err)

    # ── WS1: publish-time deflation filter (runs for every source —
    #        reserved archives or curated kept-pool — before materialise) ──
    publish_info: dict[str, Any] | None = None
    if selection_deflation == "on" and kept_ids:
        pub_book = [{"factor_id": fid, "code": prog_by_fid[fid][1].code}
                    for fid in kept_ids]
        try:
            pres = research_client.publish_book(
                pub_book, n_trials=controller.n_trials, mode=selection_deflation,
                target_horizon=target_horizon, is_frac=is_frac, val_frac=val_frac,
                cutoff_date=cutoff_date, data_dir=data_dir, n_tickers=n_tickers,
                fields=list(fields) if fields else None,
                marginal_model=marginal_model)
        except Exception as e:  # noqa: BLE001 — surfaced as a stage failure below
            pres = {"ok": False, "error": repr(e)}
        pub_kept = ([fid for fid in pres.get("kept_factor_ids", [])
                     if fid in prog_by_fid] if pres.get("ok") else [])
        if pub_kept:
            publish_info = {k: pres.get(k) for k in
                            ("passed", "combined_ic", "deflated", "n_trials",
                             "dropped", "mode") if k in pres}
            log.info("publish deflation (N_trials=%d) kept %d/%d (passed=%s, dropped=%s)",
                     controller.n_trials, len(pub_kept), len(kept_ids),
                     pres.get("passed"), pres.get("dropped"))
            kept_ids = pub_kept
        else:
            err = (pres.get("error") if not pres.get("ok")
                   else "publish deflation returned an empty book")
            if not fail_open:
                raise RuntimeError(
                    f"persist_archive: publish stage (deflation) failed: {err}. "
                    "Set QF_PERSIST_FAIL_OPEN=1 to persist the un-deflated book "
                    "instead.")
            publish_failed = True
            log.warning(
                "!!! publish deflation failed: %s — QF_PERSIST_FAIL_OPEN=1, "
                "persisting the UN-DEFLATED book (flagged in metadata)", err)

    materialised: list[tuple[EvaluatedGenome, FactorProgram, str]] = []
    for fid in kept_ids:
        eg, prog = prog_by_fid[fid]
        res = research_client.materialise_factor(prog.factor_id, prog.code)
        if res.get("ok"):
            materialised.append((eg, prog, res["code_path"]))
        else:
            log.warning("[persist:%s] materialise failed: %s",
                        prog.factor_id, res.get("error"))

    ids = [prog.factor_id for _, prog, _ in materialised]
    metrics = research_client.backtest_factors(
        factor_ids=ids, horizon=target_horizon, cutoff_date=cutoff_date,
        data_dir=data_dir, n_tickers=n_tickers) if ids else {}

    # ``model_label``/``engine`` let a non-LLM engine (e.g. the GP benchmark)
    # stamp honest provenance instead of the research-LLM defaults.
    if model_label:
        llm_model = model_label
        llm_provider = "none"
    else:
        llm_model = resolve_research_model()
        llm_provider = resolve_research_provider(llm_model)

    def _cat(raw: str) -> TradingIdeaCategory:
        try:
            return TradingIdeaCategory(raw.lower())
        except Exception:  # noqa: BLE001
            return TradingIdeaCategory.OTHER

    kept_records: list[dict[str, Any]] = []
    for eg, prog, code_path in materialised:
        m = metrics.get(prog.factor_id) or {}
        bt = (BacktestMetrics.model_validate(m["metrics"])
              if m.get("ok") else None)
        from quant_fund_agent.data.tiers import required_tier
        from quant_fund_agent.factors.registry import get_factor_class

        cls = get_factor_class(prog.factor_id)
        inputs = list(getattr(cls, "inputs", ["close"]) or ["close"]) if cls else ["close"]
        record = FactorRecord(
            id=prog.factor_id,
            name=prog.name or prog.factor_id,
            class_name="",
            description=prog.description,
            trading_idea=prog.trading_idea,
            category=_cat(prog.category),
            status=FactorStatus.BACKTESTED,
            backtest_metrics=bt,
            source_paper_ids=prog.source_paper_ids,
            source=FactorSource.RESEARCHER,
            research_session_id=session_id,
            code_path=code_path,
            required_inputs=inputs,
            required_tier=required_tier(inputs),
            prediction_horizon=prog.prediction_horizon,
            suggested_horizons=prog.suggested_horizons,
            metadata={
                "llm_model": llm_model,
                "llm_provider": llm_provider,
                "engine": engine,
                "evolution": {
                    "genome_id": eg.genome.genome_id,
                    "generation": eg.genome.generation,
                    "operator": eg.genome.operator,
                    "parent_ids": eg.genome.parent_ids,
                    "expected_sign": prog.expected_sign,
                    "objective": eg.fitness.objective.to_dict(),
                    "gates": eg.fitness.gates.to_dict(),
                    "n_trials_at_eval": eg.fitness.diagnostics.get("n_trials"),
                    "curation": curation,
                    "selection_deflation": selection_deflation,
                    "publish": publish_info,
                    **({"curation_failed": True} if curation_failed else {}),
                    **({"publish_failed": True} if publish_failed else {}),
                },
            },
        )
        kept_records.append(record.model_dump(mode="json"))

    if not kept_records:
        log.warning("persist_archive: nothing to persist (empty archive?)")
        return {"kept_factor_ids": [], "rejected_factor_ids": []}
    return research_client.persist_results(kept_records, [])
