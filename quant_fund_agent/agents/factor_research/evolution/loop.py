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
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

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
    jitter_variants,
    parse_child_response,
    random_jitter_child,
)
from quant_fund_agent.agents.factor_research.evolution.reflection import mutation_brief
from quant_fund_agent.research_eval.fitness import FitnessResult

log = logging.getLogger("evolution.loop")


@dataclass
class EvolutionRunConfig:
    """Every knob of one evolutionary research run (serialised for the record)."""

    # ── search shape ──
    generations: int = 5
    population_size: int = 10          # per island
    children_per_generation: int = 8   # candidates proposed per generation
    n_islands: int = 1
    migration_every: int = 5
    seed: int = 0
    unit: str = "single"               # "single" | "set"
    set_size: int = 3                  # initial members per SET genome

    # ── operator mix (probabilities; renormalised) ──
    p_llm_semantic: float = 0.6
    p_crossover: float = 0.25
    p_jitter: float = 0.15

    # ── seeding (generation 0, via the existing brainstorm/codegen path) ──
    n_seed_ideas: int = 8
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
    cpcv_groups: int = 6
    cpcv_k: int = 2
    embargo: int = 0
    plateau_scales: tuple[float, ...] = (0.9, 1.1)
    cutoff_date: str | None = None

    # ── fitness axes (P0+): independence basis + the crash-regime axis ──
    independence_metric: str = "residual_ic"   # "residual_ic" | "delta_participation"
    regime_kind: str = "drawdown"              # "drawdown" | "volatility"
    regime_quantile: float = 0.2
    # nonlinear LOCO combiner so conditioning/interaction value (e.g. a volatility
    # state variable) scores a positive marginal value; "ridge" = additive-only.
    marginal_model: str = "gradient_boosting"

    # ── two-stage curation (Lever 2): keep every gate-passer, curate once ──
    # "archive": the one-stage default — persist the Pareto archive (old behaviour).
    # "greedy"/"elastic_net": persist the curated kept-pool instead, so the final
    # book is not limited to the domination-pruned front.
    curation: str = "archive"
    n_keep: int | None = None                  # target book size (None → auto-sized)

    # ── data ──
    data_dir: str = "ticker_data"
    n_tickers: int | None = 15
    fixed_book: list[dict[str, Any]] = field(default_factory=list)

    # ── output ──
    out_dir: str = "data/evolution"    # overridden by the entrypoint to the scope dir

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["plateau_scales"] = list(self.plateau_scales)
        return d


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

def _codegen_program(llm: Any, idea: Any, data_context: str) -> FactorProgram | None:
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
            compile_factor(code, idea.factor_id, smoke=True)
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
                  fields: Sequence[str] | None = None) -> list[FactorProgram]:
    """Generation 0: brainstorm ideas (GraphRAG / RAG / papers / knowledge-only)
    → validated programs."""
    from quant_fund_agent.agents.factor_research.graph import _brainstorm_one
    from quant_fund_agent.agents.factor_research.state import FactorIdea, PaperSnippet
    from quant_fund_agent.mcp import research_client

    session_id = f"evo-seed-{cfg.seed}"
    brainstorm_llm = _get_llm(temperature=0.7, role="hypothesis")
    known = set(existing_ids)
    raw_ideas: list[dict] = []
    graph = None           # GraphRAG: set when the knowledge graph is in play
    mech_by_fid: dict[str, str] = {}

    if cfg.retrieval in ("rag", "graphrag"):
        # Retrieval-grounded seeding: embedding retrieval picks the papers
        # (date-gated) and the cardinality mode shapes the call(s); citations
        # come back verified.  GraphRAG additionally steers the query toward
        # computable gap mechanisms / under-covered communities and asks each
        # idea to name the mechanism it exploits (provenance link-back).
        from quant_fund_agent.knowledge.embed_store import EmbedStore
        from quant_fund_agent.knowledge.retrieval import retrieve_and_brainstorm

        focus = None
        gap_names: list[str] | None = None
        if cfg.retrieval == "graphrag":
            from quant_fund_agent.knowledge.graph_query import (
                computable_unexploited,
                island_focus,
            )
            from quant_fund_agent.knowledge.graph_store import (
                DEFAULT_GRAPH_PATH,
                KnowledgeGraph,
            )

            try:
                graph = KnowledgeGraph.load()
                gap_ids = computable_unexploited(graph, list(fields or []))
                gap_names = [graph.g.nodes[m].get("name", m)
                             for m in gap_ids][:8] or None
                focuses = island_focus(graph, max(1, cfg.n_islands),
                                       list(fields or []))
                focus = " | ".join(f for f in focuses if f) or None
                log.info("graphrag seeding: %d computable gap(s), focus=%r",
                         len(gap_ids), (focus or "")[:120])
            except FileNotFoundError:
                log.warning("knowledge graph missing (%s) — run "
                            "scripts/build_knowledge_graph.py first; falling "
                            "back to flat RAG", DEFAULT_GRAPH_PATH)
                graph = None

        store = EmbedStore()
        raw_ideas = retrieve_and_brainstorm(
            brainstorm_llm, store,
            n_ideas=cfg.n_seed_ideas, known_ids=known,
            data_context=data_context,
            cardinality=cfg.retrieval_cardinality,
            k_papers=max(cfg.rag_k, cfg.seed_papers or 0) or 4,
            cutoff_date=cfg.cutoff_date,
            focus=focus, gaps=gap_names,
            mechanism_tags=gap_names if graph is not None else None,
        )
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

        if papers:
            per_paper = max(1, -(-cfg.n_seed_ideas // len(papers)))  # ceil division
            for paper in papers:
                if len(raw_ideas) >= cfg.n_seed_ideas:
                    break
                raw_ideas.extend(_brainstorm_one(
                    brainstorm_llm, paper, per_paper, known, session_id, data_context))
        else:
            raw_ideas = _brainstorm_one(
                brainstorm_llm, None, cfg.n_seed_ideas, known, session_id, data_context)

    ideas: list[FactorIdea] = []
    seen: set[str] = set()
    for raw in raw_ideas:
        if len(ideas) >= cfg.n_seed_ideas:
            break
        idea = coerce_idea(raw, cfg.target_horizon)
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
        surviving: list[FactorIdea] = []
        for idea in ideas:
            verdict, final, _ = run_debate(
                debate_llm, _idea_payload(idea), data_context=data_context, book=[])
            if verdict == "reject":
                log.info("[seed:%s] rejected in debate", idea.factor_id)
                continue
            surviving.append(_apply_idea_revision(idea, final, cfg.target_horizon))
        log.info("debate kept %d/%d seed idea(s)", len(surviving), len(ideas))
        ideas = surviving

    codegen_llm = _get_llm(temperature=0.2, role="codegen")
    programs = []
    for idea in ideas:
        prog = _codegen_program(codegen_llm, idea, data_context)
        if prog is not None:
            programs.append(prog)
    log.info("seeded %d/%d program(s) from %d idea(s)",
             len(programs), cfg.n_seed_ideas, len(ideas))

    # GraphRAG provenance link-back: Paper → Mechanism → Factor → Field.
    if graph is not None and mech_by_fid:
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
    return programs


def coerce_idea(raw: dict[str, Any], default_horizon: int) -> "Any | None":
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
    return FactorIdea(
        factor_id=fid,
        name=raw.get("name", fid),
        category=raw.get("category", "other"),
        trading_idea=raw.get("trading_idea", ""),
        description=raw.get("description", ""),
        prediction_horizon=_coerce_horizon(raw.get("prediction_horizon"),
                                           default_horizon),
        suggested_horizons=_coerce_horizon_list(raw.get("suggested_horizons")),
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


def _apply_idea_revision(idea: Any, final: dict[str, Any], default_horizon: int) -> Any:
    """Fold a debate revision back into the idea (keeping the id if unchanged)."""
    revised = coerce_idea(final, default_horizon)
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
        self.rng = np.random.default_rng(cfg.seed)
        self.controller = EvolutionController(ControllerConfig(
            population_size=cfg.population_size,
            n_islands=cfg.n_islands,
            migration_every=cfg.migration_every,
            seed=cfg.seed,
        ))
        self.briefs: dict[str, str] = {}      # genome_id → reflection brief
        self.fields = fields                   # run-constant field set (panel cache key)
        self.data_context = data_context or self._build_data_context()
        self.out_dir = Path(cfg.out_dir)
        self.known_ids: set[str] = set()
        self.failures: list[dict[str, Any]] = []
        self._llms: dict[str, Any] = {}    # role → chat model (built lazily)
        self._program_pool: dict[str, FactorProgram] = {}  # every program ever admitted
        self.fixed_book = self._dedupe_fixed_book(cfg.fixed_book)

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
        return build_data_context(allowed, spb)

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
        fid = base
        k = 1
        taken = self.known_ids | {
            p.factor_id for eg in self.controller.population() for p in eg.genome.programs}
        while fid in taken:
            fid = f"{base}_{k}"
            k += 1
        return fid

    # ── evaluation ──

    def evaluate_program(self, program: FactorProgram) -> FitnessResult | None:
        """Score one program via the MCP seam.  Returns None on eval failure."""
        from quant_fund_agent.mcp import research_client

        book = [
            {"factor_id": b["factor_id"], "code": b["code"]}
            for b in self.fixed_book
            if b["factor_id"] != program.factor_id
        ]
        fixed_ids = {b["factor_id"] for b in book}
        book.extend(
            {"factor_id": fid, "code": code}
            for fid, code in self.controller.archive_programs()
            if fid != program.factor_id and fid not in fixed_ids
        )
        probes = [{"factor_id": pid, "code": pcode}
                  for pid, pcode in jitter_variants(program, self.cfg.plateau_scales)]

        res = research_client.evaluate_fitness(
            candidate={"factor_id": program.factor_id, "code": program.code,
                       "expected_sign": program.expected_sign},
            book=book,
            jitter=probes,
            target_horizon=self.cfg.target_horizon,
            is_frac=self.cfg.is_frac,
            val_frac=self.cfg.val_frac,
            n_trials=self.controller.n_trials + 1,  # bill this look, commit on success
            cpcv_groups=self.cfg.cpcv_groups,
            cpcv_k=self.cfg.cpcv_k,
            embargo=self.cfg.embargo,
            cutoff_date=self.cfg.cutoff_date,
            data_dir=self.cfg.data_dir,
            n_tickers=self.cfg.n_tickers,
            fields=self.fields,
            independence_metric=self.cfg.independence_metric,
            regime_kind=self.cfg.regime_kind,
            regime_quantile=self.cfg.regime_quantile,
            marginal_model=self.cfg.marginal_model,
        )
        if not res.get("ok"):
            log.info("[%s] evaluation failed: %s", program.factor_id, res.get("error"))
            self.failures.append({"factor_id": program.factor_id,
                                  "error": res.get("error")})
            return None
        self.controller.next_trial()
        return FitnessResult.from_dict(res["fitness"])

    def evaluate_set(self, programs: Sequence[FactorProgram],
                     candidate_id: str) -> FitnessResult | None:
        """SET mode: score the whole member list jointly via the MCP seam."""
        from quant_fund_agent.mcp import research_client

        res = research_client.evaluate_set_fitness(
            [{"factor_id": p.factor_id, "code": p.code} for p in programs],
            target_horizon=self.cfg.target_horizon,
            is_frac=self.cfg.is_frac,
            val_frac=self.cfg.val_frac,
            n_trials=self.controller.n_trials + 1,
            cpcv_groups=self.cfg.cpcv_groups,
            cpcv_k=self.cfg.cpcv_k,
            embargo=self.cfg.embargo,
            cutoff_date=self.cfg.cutoff_date,
            data_dir=self.cfg.data_dir,
            n_tickers=self.cfg.n_tickers,
            fields=self.fields,
            candidate_id=candidate_id,
            regime_kind=self.cfg.regime_kind,
            regime_quantile=self.cfg.regime_quantile,
            marginal_model=self.cfg.marginal_model,
        )
        if not res.get("ok"):
            log.info("[%s] set evaluation failed: %s", candidate_id, res.get("error"))
            self.failures.append({"factor_id": candidate_id,
                                  "error": res.get("error")})
            return None
        self.controller.next_trial()
        return FitnessResult.from_dict(res["fitness"])

    def _admit(self, program: FactorProgram, *, generation: int, island: int,
               operator: str, parent_ids: Sequence[str]) -> EvaluatedGenome | None:
        """SINGLE mode: wrap one program in a genome and admit it."""
        genome = Genome(
            genome_id=f"g{generation}-{program.factor_id}-{uuid.uuid4().hex[:6]}",
            programs=[program], unit="single", generation=generation,
            island=island, parent_ids=list(parent_ids), operator=operator,
        )
        return self._admit_genome(genome)

    def _admit_set(self, programs: Sequence[FactorProgram], *, generation: int,
                   island: int, operator: str,
                   parent_ids: Sequence[str]) -> EvaluatedGenome | None:
        """SET mode: wrap a member list in a genome and admit it."""
        genome = Genome(
            genome_id=f"g{generation}-set{len(programs)}-{uuid.uuid4().hex[:6]}",
            programs=list(programs), unit="set", generation=generation,
            island=island, parent_ids=list(parent_ids), operator=operator,
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
        self.briefs[genome.genome_id] = mutation_brief(
            fitness, book_size=self._book_size())
        return eg

    # ── operators ──

    def _brief_for(self, eg: EvaluatedGenome) -> str:
        return self.briefs.get(eg.genome.genome_id) or mutation_brief(
            eg.fitness, book_size=self._book_size())

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
                            ) -> tuple[FactorProgram, list[str]] | None:
        parents = self.controller.select_parents(1, island)
        if not parents:
            return None
        parent = parents[0]
        if self.cfg.debate == "on":
            child = self._child_hypothesis_debate_codegen(llm, parent)
        else:
            prompt = build_mutation_prompt(
                parent.genome.program, self._brief_for(parent),
                self.data_context, sorted(self.known_ids))
            child = self._parse_and_validate_child(llm, prompt, [parent])
        return (child, [parent.genome.genome_id]) if child is not None else None

    def _child_hypothesis_debate_codegen(self, hyp_llm: Any,
                                         parent: EvaluatedGenome,
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
            self.data_context, sorted(self.known_ids))
        try:
            raw = parse_hypothesis_response(_invoke(hyp_llm, prompt))
        except Exception as e:  # noqa: BLE001
            log.info("hypothesis response unparsable: %s", e)
            return None

        verdict, final, _ = run_debate(
            self._role_llm("debate", 0.3), raw,
            data_context=self.data_context, book=self._book_entries())
        if verdict == "reject":
            log.info("[%s] child hypothesis rejected in debate", raw.get("factor_id"))
            return None

        idea = coerce_idea(final, self.cfg.target_horizon)
        if idea is None:
            return None
        idea.factor_id = self._unique_id(idea.factor_id)
        prog = _codegen_program(self._role_llm("codegen", 0.2), idea,
                                self.data_context)
        if prog is not None and not prog.source_paper_ids:
            prog.source_paper_ids = list(parent.genome.program.source_paper_ids)
        return prog

    def _child_crossover(self, llm: Any, island: int,
                         ) -> tuple[FactorProgram, list[str]] | None:
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
        prompt = build_crossover_prompt(
            a.genome.program, self._brief_for(a),
            b.genome.program, self._brief_for(b),
            self.data_context, sorted(self.known_ids))
        child = self._parse_and_validate_child(llm, prompt, [a, b])
        if child is None:
            return None
        if self.cfg.debate == "on":
            # Post-hoc accept/reject only (the code already exists, so a
            # "revise" verdict can't be honoured without regenerating it).
            from quant_fund_agent.agents.factor_research.debate import run_debate

            verdict, _, _ = run_debate(
                self._role_llm("debate", 0.3), _idea_payload(child),
                data_context=self.data_context, book=self._book_entries(),
                max_revisions=0)
            if verdict == "reject":
                log.info("[%s] crossover child rejected in debate", child.factor_id)
                return None
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
                fid = self._unique_id(program.factor_id)
                if fid != program.factor_id:
                    program.code = rewrite_factor_id(program.code,
                                                     program.factor_id, fid)
                    program.factor_id = fid
                compile_factor(program.code, program.factor_id, smoke=True)
                break
            except Exception as e:  # noqa: BLE001 — feed the error back once
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

    def _child_jitter(self, island: int) -> tuple[FactorProgram, list[str]] | None:
        parents = self.controller.select_parents(1, island)
        if not parents:
            return None
        parent = parents[0]
        new_id = self._unique_id(f"{parent.genome.program.factor_id}_j")
        child = random_jitter_child(parent.genome.program, self.rng, new_id)
        return (child, [parent.genome.genome_id]) if child is not None else None

    # ── SET-mode operators (structural; deterministic given the seed) ──

    def _child_set(self, op: str, island: int,
                   ) -> tuple[list[FactorProgram], list[str]] | None:
        """One SET child via ``structural`` (add/drop/replace member),
        ``splice`` (union-sample two parents) or ``member_jitter``.

        All three are programmatic — the LLM's creativity enters SET mode
        through the generation-0 members (and any future member-level LLM
        mutation, a documented extension); the set-level search explores
        *composition* space, which is what the unit is for.
        """
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

        # structural: add / drop / replace against the run-wide member pool
        current = set(a.genome.factor_ids)
        external = [p for fid, p in sorted(self._program_pool.items())
                    if fid not in current]
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
        with (self.out_dir / "lineage.jsonl").open("w") as fh:
            for row in self.controller.lineage:
                fh.write(json.dumps(row, default=str) + "\n")

    # ── main drive ──

    def run(self, initial_programs: Sequence[FactorProgram] | None = None,
            ) -> dict[str, Any]:
        """Run the whole evolutionary search; returns a summary dict.

        ``initial_programs`` (mostly for tests / resumed runs) bypasses the LLM
        seed brainstorm.
        """
        cfg = self.cfg
        t0 = time.time()
        self._load_known_ids()

        # ── generation 0: seed ──
        programs = list(initial_programs) if initial_programs is not None else \
            seed_programs(cfg, self.data_context, self.known_ids, self.fields)
        for prog in programs:  # the pool SET-structural ops draw from
            self._program_pool.setdefault(prog.factor_id, prog)
        if cfg.unit == "set":
            # Partition the seed pool into initial sets of ~set_size (round-robin
            # chunks, so every seed program appears in exactly one set).
            size = max(1, cfg.set_size)
            for k in range(0, len(programs), size):
                chunk = programs[k:k + size]
                if chunk:
                    self._admit_set(chunk, generation=0,
                                    island=(k // size) % max(1, cfg.n_islands),
                                    operator="seed", parent_ids=[])
        else:
            for k, prog in enumerate(programs):
                self._admit(prog, generation=0, island=k % max(1, cfg.n_islands),
                            operator="seed", parent_ids=[])
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
                ("member_jitter", max(0.0, cfg.p_jitter)),
            ]
        else:
            ops = [
                ("llm_semantic", max(0.0, cfg.p_llm_semantic)),
                ("crossover", max(0.0, cfg.p_crossover)),
                ("jitter", max(0.0, cfg.p_jitter)),
            ]
        total_p = sum(p for _, p in ops) or 1.0
        op_names = [n for n, _ in ops]
        op_probs = [p / total_p for _, p in ops]

        for gen in range(1, cfg.generations + 1):
            self.controller.generation = gen
            made = 0
            for k in range(cfg.children_per_generation):
                island = k % max(1, cfg.n_islands)
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
                if op == "jitter":
                    made_child = self._child_jitter(island)
                elif op == "crossover":
                    made_child = self._child_crossover(
                        self._role_llm("hypothesis", 0.6), island)
                else:
                    made_child = self._child_llm_semantic(
                        self._role_llm("hypothesis", 0.6), island)
                if made_child is None:
                    continue
                child, parent_ids = made_child
                eg = self._admit(child, generation=gen, island=island,
                                 operator=op, parent_ids=parent_ids)
                if eg is not None:
                    made += 1
            if cfg.n_islands > 1 and gen % max(1, cfg.migration_every) == 0:
                moved = self.controller.migrate()
                log.info("generation %d: migrated %d elite(s)", gen, moved)
            self._checkpoint()
            log.info("generation %d: %d/%d children admitted; archive=%d; n_trials=%d",
                     gen, made, cfg.children_per_generation,
                     len(self.controller.archive), self.controller.n_trials)

        return self.summary(time.time() - t0)

    def summary(self, elapsed: float) -> dict[str, Any]:
        return {
            "generations": self.controller.generation,
            "n_trials": self.controller.n_trials,
            "population": len(self.controller.population()),
            "fixed_book_size": len(self.fixed_book),
            "archive": [
                {"factor_ids": eg.genome.factor_ids,
                 "genome_id": eg.genome.genome_id,
                 "unit": eg.genome.unit,
                 "operator": eg.genome.operator,
                 "generation": eg.genome.generation,
                 "objective": eg.fitness.objective.to_dict()}
                for eg in self.controller.archive
            ],
            "n_eval_failures": len(self.failures),
            "elapsed_sec": round(elapsed, 1),
        }


# ── persisting the survivors ──────────────────────────────────────────────────

def persist_archive(controller: EvolutionController, *, session_id: str,
                    target_horizon: int, cutoff_date: str | None = None,
                    data_dir: str = "ticker_data", n_tickers: int | None = 15,
                    curation: str = "archive", n_keep: int | None = None,
                    is_frac: float = 0.6, val_frac: float = 0.2,
                    fields: Sequence[str] | None = None,
                    marginal_model: str = "gradient_boosting",
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

    # ── choose the source book (archive vs the full kept-pool) ──
    use_pool = curation != "archive" and bool(controller.kept_pool)
    source = controller.kept_pool if use_pool else controller.archive
    prog_by_fid: dict[str, tuple[EvaluatedGenome, FactorProgram]] = {}
    for eg in source:
        for prog in eg.genome.programs:
            prog_by_fid.setdefault(prog.factor_id, (eg, prog))

    kept_ids = list(prog_by_fid)
    curation_info: dict[str, Any] | None = None
    if use_pool and prog_by_fid:
        pool = [{"factor_id": fid, "code": prog.code}
                for fid, (_, prog) in prog_by_fid.items()]
        res = research_client.curate_book(
            pool, mode=curation, n_keep=n_keep, target_horizon=target_horizon,
            is_frac=is_frac, val_frac=val_frac, cutoff_date=cutoff_date,
            data_dir=data_dir, n_tickers=n_tickers,
            fields=list(fields) if fields else None, marginal_model=marginal_model)
        if res.get("ok"):
            kept_ids = [fid for fid in res.get("kept_factor_ids", [])
                        if fid in prog_by_fid] or kept_ids
            curation_info = {k: res[k] for k in
                             ("mode", "combined_ic", "selection_frequency", "n_pool")
                             if k in res}
            log.info("curation (%s) kept %d/%d factors from the pool",
                     curation, len(kept_ids), len(prog_by_fid))
        else:
            log.warning("curation (%s) failed: %s — persisting the full pool",
                        curation, res.get("error"))

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
                "engine": "evolution",
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
                },
            },
        )
        kept_records.append(record.model_dump(mode="json"))

    if not kept_records:
        log.warning("persist_archive: nothing to persist (empty archive?)")
        return {"kept_factor_ids": [], "rejected_factor_ids": []}
    return research_client.persist_results(kept_records, [])
