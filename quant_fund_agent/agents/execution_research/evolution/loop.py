"""The execution-arm evolution loop (E1: seeds + param-jitter, no LLM).

Reuses the factor arm's ``EvolutionController`` verbatim (constrained NSGA-II,
Pareto archive, N_trials billing, islands, lineage, save/load) — exactly the
GP-benchmark reuse boundary — and swaps only:

* the program type (``ExecutionProgram`` via ``Genome(program_type="executor")``),
* the child operators (E1: ``random_jitter_child`` over the ``params`` dict;
  E2 adds LLM-semantic mutation), and
* the evaluation call (``research_client.evaluate_executor_fitness`` against a
  ``FrozenSignalSet`` manifest — in-process under ``QF_USE_MCP=0``).

Block API (consumed by the joint outer layer, `docs/joint-evolution/DESIGN.md`):
``run(resume=True, n_generations=G)`` continues from ``out_dir/state.json``;
``sota_executor()`` is the frozen-SOTA view the factor arm's coupling seam
receives; ``rescore_archive(manifest)`` deterministically re-scores every held
genome after a re-freeze (bills the joint ledger's *look* count at the outer
layer — never ``n_trials`` here).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from quant_fund_agent.agents.execution_research.evolution.genome import (
    ExecutionProgram,
)
from quant_fund_agent.agents.execution_research.evolution.mutation import (
    build_exec_crossover_prompt,
    build_exec_mutation_prompt,
    jitter_variants,
    parse_exec_child_response,
    random_jitter_child,
)
from quant_fund_agent.agents.execution_research.evolution.reflection import (
    exec_mutation_brief,
)
from quant_fund_agent.agents.execution_research.evolution.seeds import (
    seed_execution_programs,
)
from quant_fund_agent.agents.factor_research.evolution.controller import (
    ControllerConfig,
    EvaluatedGenome,
    EvolutionController,
)
from quant_fund_agent.agents.factor_research.evolution.genome import Genome
# the per-role chat-LLM factory + invoke helper are shared with the factor arm
from quant_fund_agent.agents.factor_research.evolution.loop import (
    _get_llm,
    _invoke,
)
from quant_fund_agent.research_eval.fitness import FitnessResult

log = logging.getLogger("execution_research.loop")


@dataclass
class ExecEvolutionRunConfig:
    """Every knob of one execution-evolution run (JSON-serialisable)."""

    name: str = "exec-evolution"
    out_dir: str = "data/evolution_exec"
    signals_manifest: str = ""            # FrozenSignalSet manifest path (required)
    generations: int = 5
    population_size: int = 8
    children_per_generation: int = 6
    n_islands: int = 1
    migration_every: int = 5
    seed: int = 0
    regime: str | None = None             # None = seed both book shapes
    jitter_pct: float = 0.15
    plateau_scales: tuple[float, ...] = (0.9, 1.1)
    # ── operator mix (E2): defaults keep the E1 jitter-only run byte-identical ──
    p_llm_semantic: float = 0.0
    p_crossover: float = 0.0
    p_jitter: float = 1.0
    # ── E3: skeptic debate + RAG grounding (both fail open; ablation arms) ──
    debate: str = "off"                 # "on" | "off"
    retrieval: str = "none"             # "none" | "rag" 
    # ── evaluation knobs (threaded to the service; held fixed across a run) ──
    is_frac: float = 0.6
    val_frac: float = 0.2
    cutoff_date: str | None = None
    data_dir: str = "ticker_data"
    n_tickers: int | None = 15
    fields: list[str] | None = None
    cost_rate: float = 5e-4
    lambda_dispersion: float = 0.5
    gate_turnover: float | None = None
    gate_degradation: float = 0.5
    min_activity: float = 0.05
    selection_deflation: str = "off"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["plateau_scales"] = list(self.plateau_scales)
        return d


class ExecEvolutionLoop:
    """Drive select → jitter-mutate → evaluate → insert for executor programs."""

    def __init__(self, cfg: ExecEvolutionRunConfig,
                 controller: EvolutionController | None = None) -> None:
        if not cfg.signals_manifest:
            raise ValueError("ExecEvolutionRunConfig.signals_manifest is required "
                             "(freeze the evaluation signals first)")
        self.cfg = cfg
        self.out_dir = Path(cfg.out_dir)
        self.controller = controller or EvolutionController(ControllerConfig(
            population_size=cfg.population_size,
            n_islands=cfg.n_islands,
            migration_every=cfg.migration_every,
            seed=cfg.seed,
        ))
        self.rng = np.random.default_rng(cfg.seed)
        self.failures: list[dict[str, Any]] = []
        self._child_counter = 0
        self.briefs: dict[str, str] = {}      # genome_id → deterministic brief
        self._llms: dict[str, Any] = {}       # role → cached chat LLM
        self._rag_block: str | None = None    # E3: cached literature snippets
        self.debate_transcripts: list[dict[str, Any]] = []

    # ── evaluation (the reward channel) ───────────────────────────────────────

    def _archive_payload(self, exclude_id: str) -> list[dict[str, str]]:
        """Archived executor programs → the structural-novelty comparators."""
        out = []
        for eg in self.controller.archive:
            for p in eg.genome.programs:
                if p.factor_id != exclude_id:
                    out.append({"executor_id": p.factor_id, "code": p.code})
        return out

    def evaluate_program(self, program: ExecutionProgram,
                         *, bill: bool = True) -> FitnessResult | None:
        """Score one executor via the MCP seam.  Returns None on eval failure.

        ``bill=False`` is the re-score path (a fresh look at VAL against new
        frozen signals, but NOT a new hypothesis — the joint ledger bills it as
        a look at the outer layer; ``n_trials`` here must not move).
        """
        from quant_fund_agent.mcp import research_client

        probes = [{"executor_id": pid, "code": pcode}
                  for pid, pcode in jitter_variants(program, self.cfg.plateau_scales)]
        n_trials = self.controller.n_trials + 1 if bill else max(
            1, self.controller.n_trials)
        res = research_client.evaluate_executor_fitness(
            candidate={"executor_id": program.executor_id, "code": program.code},
            signals_manifest=self.cfg.signals_manifest,
            jitter=probes or None,
            archive=self._archive_payload(program.executor_id) or None,
            n_trials=n_trials,
            is_frac=self.cfg.is_frac,
            val_frac=self.cfg.val_frac,
            cutoff_date=self.cfg.cutoff_date,
            data_dir=self.cfg.data_dir,
            n_tickers=self.cfg.n_tickers,
            fields=self.cfg.fields,
            cost_rate=self.cfg.cost_rate,
            lambda_dispersion=self.cfg.lambda_dispersion,
            gate_turnover=self.cfg.gate_turnover,
            gate_degradation=self.cfg.gate_degradation,
            min_activity=self.cfg.min_activity,
            selection_deflation=self.cfg.selection_deflation,
        )
        if not res.get("ok"):
            log.info("[%s] evaluation failed: %s", program.executor_id,
                     res.get("error"))
            self.failures.append({"executor_id": program.executor_id,
                                  "error": res.get("error")})
            return None
        if bill:
            self.controller.next_trial()
        return FitnessResult.from_dict(res["fitness"])

    def _admit(self, program: ExecutionProgram, *, generation: int, island: int,
               operator: str, parent_ids: Sequence[str]) -> EvaluatedGenome | None:
        genome = Genome(
            genome_id=f"x{generation}-{program.executor_id}-{uuid.uuid4().hex[:6]}",
            programs=[program], unit="single", generation=generation,
            island=island, parent_ids=list(parent_ids), operator=operator,
            program_type="executor",
        )
        if self.controller.is_duplicate(genome):
            log.info("[%s] duplicate genome — skipped (not billed)",
                     program.executor_id)
            return None
        fitness = self.evaluate_program(program)
        if fitness is None:
            return None
        eg = EvaluatedGenome(genome=genome, fitness=fitness)
        self.controller.insert(eg)
        # the deterministic teacher brief the mutating LLM sees (E2)
        self.briefs[genome.genome_id] = exec_mutation_brief(fitness)
        return eg

    # ── checkpoint (same trio as the factor loop) ─────────────────────────────

    def _checkpoint(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.controller.save(self.out_dir / "state.json")
        (self.out_dir / "run_config.json").write_text(
            json.dumps(self.cfg.to_dict(), indent=2))
        with (self.out_dir / "lineage.jsonl").open("w") as fh:
            for row in self.controller.lineage:
                fh.write(json.dumps(row, default=str) + "\n")

    # ── main drive ────────────────────────────────────────────────────────────

    def run(self, initial_programs: Sequence[ExecutionProgram] | None = None, *,
            resume: bool = False, n_generations: int | None = None,
            ) -> dict[str, Any]:
        """Seed (or resume) → jitter-mutate → evaluate → insert, G generations."""
        cfg = self.cfg
        t0 = time.time()
        self._n_trials_at_entry = 0   # block accounting (joint layer)

        if resume:
            state_path = self.out_dir / "state.json"
            self.controller = EvolutionController.load(state_path)
            self._n_trials_at_entry = self.controller.n_trials
            lineage_path = self.out_dir / "lineage.jsonl"
            if lineage_path.exists():
                self.controller.lineage = [
                    json.loads(line)
                    for line in lineage_path.read_text().splitlines() if line.strip()
                ]
            log.info("resumed from %s: generation=%d, archive=%d, n_trials=%d",
                     state_path, self.controller.generation,
                     len(self.controller.archive), self.controller.n_trials)
        else:
            programs = (list(initial_programs) if initial_programs is not None
                        else seed_execution_programs())
            if cfg.regime:
                programs = [p for p in programs if p.regime == cfg.regime]
            for k, prog in enumerate(programs):
                self._admit(prog, generation=0, island=k % max(1, cfg.n_islands),
                            operator="seed", parent_ids=[])
            self._checkpoint()
            log.info("generation 0: population=%d, archive=%d, n_trials=%d",
                     len(self.controller.population()),
                     len(self.controller.archive), self.controller.n_trials)
        if not self.controller.population():
            log.warning("empty seed population — aborting run")
            return self.summary(time.time() - t0)

        ops = [("llm_semantic", max(0.0, cfg.p_llm_semantic)),
               ("crossover", max(0.0, cfg.p_crossover)),
               ("jitter", max(0.0, cfg.p_jitter))]
        total_p = sum(p for _, p in ops) or 1.0
        op_names = [n for n, _ in ops]
        op_probs = [p / total_p for _, p in ops]

        start_gen = self.controller.generation + 1
        end_gen = self.controller.generation + (
            n_generations if n_generations is not None else cfg.generations)
        for gen in range(start_gen, end_gen + 1):
            self.controller.generation = gen
            made = 0
            for k in range(cfg.children_per_generation):
                island = k % max(1, cfg.n_islands)
                op = str(self.rng.choice(op_names, p=op_probs))
                if op == "jitter":
                    made_child = self._child_jitter(gen, island)
                elif op == "crossover":
                    made_child = self._child_crossover(gen, island)
                else:
                    made_child = self._child_llm_semantic(gen, island)
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
            log.info("generation %d: %d/%d children admitted; archive=%d; "
                     "n_trials=%d", gen, made, cfg.children_per_generation,
                     len(self.controller.archive), self.controller.n_trials)
        return self.summary(time.time() - t0)

    # ── child operators ───────────────────────────────────────────────────────

    def _new_child_id(self, gen: int) -> str:
        self._child_counter += 1
        return f"ex{gen}_{self._child_counter}_{uuid.uuid4().hex[:4]}"

    def _brief_for(self, eg: EvaluatedGenome) -> str:
        return self.briefs.get(eg.genome.genome_id) \
            or exec_mutation_brief(eg.fitness)

    def _known_ids(self) -> list[str]:
        out: set[str] = set()
        for pool in (*self.controller.islands, self.controller.archive,
                     self.controller.kept_pool):
            for eg in pool:
                out.update(eg.genome.factor_ids)
        return sorted(out)

    def _role_llm(self, role: str, temperature: float) -> Any:
        if role not in self._llms:
            self._llms[role] = _get_llm(temperature=temperature, role=role)
        return self._llms[role]

    def _child_jitter(self, gen: int, island: int,
                      ) -> tuple[ExecutionProgram, list[str]] | None:
        parents = self.controller.select_parents(1, island)
        if not parents:
            return None
        parent = parents[0]
        child = random_jitter_child(parent.genome.program, self.rng,
                                    self._new_child_id(gen), self.cfg.jitter_pct)
        return (child, [parent.genome.genome_id]) if child is not None else None

    def _rag_snippets(self) -> str:
        """E3: execution-literature snippets, retrieved once per loop."""
        if self.cfg.retrieval != "rag":
            return ""
        if self._rag_block is None:
            from quant_fund_agent.agents.execution_research.evolution.debate import (
                execution_literature_snippets,
            )
            self._rag_block = execution_literature_snippets()
        return self._rag_block

    def _archive_mechanisms(self) -> list[str]:
        out = []
        for eg in self.controller.archive:
            m = getattr(eg.genome.program, "mechanism", "")
            if m:
                out.append(m)
        return out

    def _parse_and_validate_child(self, llm: Any, prompt: str,
                                  expected_id: str) -> ExecutionProgram | None:
        """LLM call → JSON parse → [debate] → static validation + smoke → program.

        Every failure is logged and skipped (a bad generation is a data point);
        nothing an LLM emits reaches evaluation without passing the same
        validator the persist path runs.  With ``debate="on"`` the skeptic
        attacks the proposal BEFORE it costs an evaluation (≤1 revision; the
        possibly-revised code re-enters validation; rejects are dropped).
        """
        from quant_fund_agent.execution.codegen import (
            ExecutorValidationError,
            compile_executor_inmem,
        )

        try:
            payload = parse_exec_child_response(_invoke(llm, prompt))
        except Exception as e:  # noqa: BLE001
            log.info("child response unparsable: %s", e)
            return None
        if self.cfg.debate == "on":
            from quant_fund_agent.agents.execution_research.evolution.debate import (
                run_exec_debate,
            )
            verdict, payload, transcript = run_exec_debate(
                self._role_llm("debate", 0.3), payload,
                archive_mechanisms=self._archive_mechanisms())
            self.debate_transcripts.append(
                {"executor_id": expected_id, "verdict": verdict,
                 "transcript": transcript})
            if verdict == "reject":
                log.info("[%s] child rejected in debate", expected_id)
                return None
        if payload["executor_id"] != expected_id:
            log.info("child used id %r instead of %r — rewriting",
                     payload["executor_id"], expected_id)
            payload["code"] = payload["code"].replace(
                payload["executor_id"], expected_id)
            payload["executor_id"] = expected_id
        try:
            compile_executor_inmem(payload["code"], expected_id)
        except ExecutorValidationError as e:
            log.info("[%s] child failed validation/smoke: %s", expected_id, e)
            return None
        except Exception as e:  # noqa: BLE001
            log.info("[%s] child failed to compile: %s", expected_id, e)
            return None
        return ExecutionProgram(
            executor_id=expected_id,
            code=payload["code"],
            name=payload.get("name", expected_id),
            regime=payload.get("regime", "per_underlying"),
            mechanism=payload.get("mechanism", ""),
            expected_effect=payload.get("expected_effect", ""),
        )

    def _child_llm_semantic(self, gen: int, island: int,
                            ) -> tuple[ExecutionProgram, list[str]] | None:
        parents = self.controller.select_parents(1, island)
        if not parents:
            return None
        parent = parents[0]
        new_id = self._new_child_id(gen)
        prompt = build_exec_mutation_prompt(
            parent.genome.program, self._brief_for(parent), new_id,
            self._known_ids())
        rag = self._rag_snippets()
        if rag:
            prompt = rag + "\n\n" + prompt
        child = self._parse_and_validate_child(
            self._role_llm("codegen", 0.4), prompt, new_id)
        return (child, [parent.genome.genome_id]) if child is not None else None

    def _child_crossover(self, gen: int, island: int,
                         ) -> tuple[ExecutionProgram, list[str]] | None:
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
        new_id = self._new_child_id(gen)
        prompt = build_exec_crossover_prompt(
            a.genome.program, self._brief_for(a),
            b.genome.program, self._brief_for(b), new_id, self._known_ids())
        child = self._parse_and_validate_child(
            self._role_llm("codegen", 0.4), prompt, new_id)
        if child is None:
            return None
        return child, [a.genome.genome_id, b.genome.genome_id]

    # ── the block/joint-layer seams ───────────────────────────────────────────

    def sota_executor(self) -> dict[str, Any] | None:
        """The frozen-SOTA view of the archive (DESIGN §Block interface).

        Gate-passing archive member with the maximum primary axis (net deflated
        VAL Sharpe); ties → cost efficiency (robustness slot) → parsimony →
        lowest ``genome_id`` (fully deterministic).  ``None`` until something
        passes the gates.
        """
        def _v(x: float | None) -> float:
            return float("-inf") if x is None else float(x)

        candidates = [eg for eg in self.controller.archive if eg.selectable]
        if not candidates:
            return None
        best = sorted(
            candidates,
            key=lambda eg: (-_v(eg.fitness.objective.marginal_value),
                            -_v(eg.fitness.objective.robustness),
                            -_v(eg.fitness.objective.parsimony),
                            eg.genome.genome_id),
        )[0]
        prog = best.genome.program
        return {
            "executor_id": prog.factor_id,
            "code": prog.code,
            "regime": getattr(prog, "regime", "per_underlying"),
            "genome_id": best.genome.genome_id,
            "objective": best.fitness.objective.to_dict(),
        }

    def rescore_archive(self, signals_manifest: str) -> dict[str, Any]:
        """Deterministically re-score every held genome against new frozen signals.

        Called by the joint layer after a factor-block re-freeze.  Every unique
        genome across islands / archive / kept_pool is re-evaluated against the
        new ``FrozenSignalSet`` (``n_trials`` untouched — the outer ledger
        bills these as *looks*), fitness is replaced in place, and the archive
        + kept_pool are rebuilt so domination and gate structure reflect the
        new signals.  A genome whose re-score *fails* keeps its old fitness and
        is reported (never silently dropped).
        """
        self.cfg = replace(self.cfg, signals_manifest=signals_manifest)

        instances: dict[str, list[EvaluatedGenome]] = {}
        for pool in (*self.controller.islands, self.controller.archive,
                     self.controller.kept_pool):
            for eg in pool:
                instances.setdefault(eg.genome.genome_id, []).append(eg)

        rescored, failed = 0, 0
        for gid, egs in instances.items():
            fitness = self.evaluate_program(egs[0].genome.program, bill=False)
            if fitness is None:
                failed += 1
                continue
            for eg in egs:
                eg.fitness = fitness
            rescored += 1

        # Rebuild the derived structures under the new fitness landscape:
        # kept_pool = gate-passers (per its contract), archive = their
        # non-dominated front, both re-derived from everything we hold.
        unique = {gid: egs[0] for gid, egs in instances.items()}
        survivors = [eg for eg in unique.values() if eg.selectable]
        self.controller.kept_pool = list(survivors)
        self.controller._pool_fingerprints = {
            eg.genome.code_fingerprint() for eg in survivors}
        self.controller.archive = []
        for eg in survivors:
            self.controller._update_archive(eg)
        self.controller.lineage.append({
            "event": "rescore_archive",
            "signals_manifest": signals_manifest,
            "rescored": rescored, "failed": failed,
            "archive_after": len(self.controller.archive),
            "generation": self.controller.generation,
        })
        self._checkpoint()
        return {"rescored": rescored, "failed": failed,
                "archive": len(self.controller.archive),
                "kept_pool": len(self.controller.kept_pool),
                "n_looks": rescored}

    def summary(self, elapsed: float) -> dict[str, Any]:
        return {
            "generations": self.controller.generation,
            "n_trials": self.controller.n_trials,
            "population": len(self.controller.population()),
            "archive": [
                {"executor_ids": eg.genome.factor_ids,
                 "genome_id": eg.genome.genome_id,
                 "operator": eg.genome.operator,
                 "generation": eg.genome.generation,
                 "objective": eg.fitness.objective.to_dict()}
                for eg in self.controller.archive
            ],
            "sota_executor": (self.sota_executor() or {}).get("executor_id"),
            "n_eval_failures": len(self.failures),
            "elapsed_sec": round(elapsed, 1),
        }
