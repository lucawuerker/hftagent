"""The GP factor-mining loop.

Reuses the LLM arm's deterministic machinery verbatim — the NSGA-II
``EvolutionController`` (selection, Pareto archive, N_trials/deflation, islands,
lineage and checkpoint/resume), the ``evaluate_fitness`` scoring
seam, and ``persist_archive`` — and swaps only the *operator layer* for
deterministic GP:

* **seed** — random typed trees (ramped half-and-half);
* **mutate/recombine** — subtree crossover, subtree/point/hoist mutation;
* **hierarchical growth** (AutoAlpha) — a depth schedule that raises the tree
  depth cap in stages, carrying the archive across stages so deeper factors
  recombine shallow winners.

Because the controller and scoring are identical to the LLM arm, a GP run is a
clean apples-to-apples benchmark: only *how children are proposed* differs.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

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
from quant_fund_agent.agents.factor_research.gp import operators
from quant_fund_agent.agents.factor_research.gp.grammar import (
    DEFAULT_CONSTS,
    DEFAULT_WINDOWS,
    Node,
    build_grammar,
    depth,
    random_tree,
)
from quant_fund_agent.agents.factor_research.gp.render import tree_to_program
from quant_fund_agent.research_eval.fitness import FitnessResult

log = logging.getLogger("factor_research.gp")


@dataclass
class GPRunConfig:
    """Every knob of one GP benchmark run (serialised into the record)."""

    # ── search shape ──
    generations: int = 5                # generations PER depth stage
    population_size: int = 20           # per island
    children_per_generation: int = 16
    n_islands: int = 1
    migration_every: int = 5
    seed: int = 0
    seed_pop: int = 40                  # gen-0 random trees

    # ── hierarchical growth (AutoAlpha) ──
    # Max op-nesting depth per stage; each stage runs ``generations`` gens with
    # the archive carried across (deeper stages recombine shallow winners).
    depth_schedule: tuple[int, ...] = (3, 5)

    # ── GP operator mix (probabilities; renormalised) ──
    p_crossover: float = 0.5
    p_subtree: float = 0.25
    p_point: float = 0.15
    p_hoist: float = 0.10

    # ── grammar knobs ──
    window_pool: tuple[int, ...] = DEFAULT_WINDOWS
    const_pool: tuple[float, ...] = DEFAULT_CONSTS
    category: str = "statistical_arbitrage"
    max_resample_tries: int = 12        # attempts to find a valid, non-degenerate child

    # ── evaluation (threads straight into research_eval, same as the LLM arm) ──
    target_horizon: int = 6
    is_frac: float = 0.6
    val_frac: float = 0.2
    stability_blocks: int = 4
    plateau_scales: tuple[float, ...] = (0.9, 1.1)
    cutoff_date: str | None = None

    # ── progressive data reveal (ported from the LLM arm 2026-08-09 so the GP
    # baseline can run the SAME two-phase walk-forward protocol; default OFF =
    # byte-identical legacy behaviour).  ``generations`` here means generations
    # per depth stage — the reveal schedule spans the TOTAL generation count
    # (len(depth_schedule) * generations). ──
    # Stationarity gate (2026-08-09): reject candidates whose harness
    # ``level_rho`` diagnostic exceeds this (None = off).  Motivated by the
    # ungated L0WF run gaming the per-underlying IC with ``log(high)``.
    level_rho_max: float | None = None

    progressive: bool = False
    test_frac: float = 0.2       # final never-revealed TEST tail
    seed_frac: float = 0.45      # fraction of DEV visible at generation 0
    reveal_every: int = 1        # generations between reveals
    val_blocks: int = 2          # sliding VAL = last val_blocks blocks
    wf_blocks: int = 0           # two-phase WF: last wf_blocks gens reveal one
    wf_block_bars: int = 126     # fixed block each, prequentially scored first

    # ── fitness axes ──
    independence_metric: str = "residual_ic"
    marginal_model: str = "gradient_boosting"

    # ── economic realism (all default OFF, matching the baseline arm) ──
    gate_turnover: float | None = None
    cost_rate: float = 5e-4
    perturbation_weight: float = 0.0
    perturbation_sigma: float = 0.5

    # ── two-stage curation / publish deflation (used at persist) ──
    curation: str = "archive"
    n_keep: int | None = None
    selection_deflation: str = "off"

    # ── data ──
    data_dir: str = "ticker_data"
    n_tickers: int | None = 15
    reference_book: list[dict[str, Any]] = field(default_factory=list)

    # ── output ──
    out_dir: str = "data/gp"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("depth_schedule", "window_pool", "const_pool", "plateau_scales"):
            d[k] = list(getattr(self, k))
        return d


class GPLoop:
    """Drives one GP factor-mining run."""

    def __init__(self, cfg: GPRunConfig, allowed_fields: Sequence[str] | None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.controller = EvolutionController(ControllerConfig(
            population_size=cfg.population_size,
            n_islands=cfg.n_islands,
            migration_every=cfg.migration_every,
            seed=cfg.seed,
        ))
        self.fields = sorted(allowed_fields) if allowed_fields else None
        self.grammar = build_grammar(
            allowed_fields, window_pool=cfg.window_pool, const_pool=cfg.const_pool)
        self.fixed_book: list[dict[str, Any]] = []
        self.reference_book = list(cfg.reference_book or [])
        self.out_dir = Path(cfg.out_dir)
        self.known_ids: set[str] = set()
        self.failures: list[dict[str, Any]] = []
        self._trees: dict[str, Node] = {}     # factor_id → source tree
        self._counter = 0
        self._synth_panel: dict[str, Any] | None = None
        # progressive reveal state (None ⇒ legacy fixed-split behaviour)
        self._schedule = None
        self._window = None
        self._prequential_done: set[int] | None = None

    # ── id / known-factor bookkeeping (mirrors EvolutionLoop) ──

    def _load_known_ids(self) -> None:
        from quant_fund_agent.mcp import research_client
        try:
            self.known_ids = set(research_client.existing_factor_ids(scope="package"))
        except Exception as e:  # noqa: BLE001
            log.warning("could not load existing factor ids (%s)", e)
            self.known_ids = set()

    def _fresh_id(self) -> str:
        base = f"gp_{self._counter:04d}"
        self._counter += 1
        fid = base
        k = 1
        taken = self.known_ids | {
            p.factor_id for eg in self.controller.population()
            for p in eg.genome.programs}
        while fid in taken:
            fid = f"{base}_{k}"
            k += 1
        return fid

    # ── tree ↔ genome round-trip ──

    def _tree_for(self, eg: EvaluatedGenome) -> Node | None:
        fid = eg.genome.program.factor_id
        tree = self._trees.get(fid)
        if tree is not None:
            return tree
        raw = eg.genome.metadata.get("gp_tree")
        if raw:
            tree = Node.from_dict(raw)
            self._trees[fid] = tree
            return tree
        return None

    # ── candidate validity: compiles + non-degenerate on synthetic data ──

    def _acceptable(self, program: FactorProgram) -> bool:
        from quant_fund_agent.agents.factor_research.codegen import _make_synthetic_panel
        from quant_fund_agent.factors import inmem

        if self._synth_panel is None:
            self._synth_panel = _make_synthetic_panel()
        try:
            sig = inmem.signal_from_code(
                program.code, program.factor_id, self._synth_panel)
        except Exception as e:  # noqa: BLE001 — invalid / raising factor
            self.failures.append({"factor_id": program.factor_id, "error": str(e)})
            return False
        arr = np.asarray(sig.to_numpy(), dtype="float64")
        finite = np.isfinite(arr)
        if finite.sum() < max(5, int(0.02 * arr.size)):
            return False
        vals = arr[finite]
        return vals.size > 0 and float(np.std(vals)) > 1e-12

    def _is_duplicate(self, program: FactorProgram) -> bool:
        probe = Genome(genome_id="probe", programs=[program], unit="single")
        return self.controller.is_duplicate(probe)

    # ── child proposal ──

    def _seed_one(self, max_depth: int, method: str) -> tuple[FactorProgram, Node] | None:
        for _ in range(self.cfg.max_resample_tries):
            tree = random_tree(self.grammar, self.rng, max_depth, method=method)
            program = self._render(tree)
            if self._is_duplicate(program) or not self._acceptable(program):
                continue
            return program, tree
        return None

    def _propose(self, op_name: str, island: int, max_depth: int,
                 ) -> tuple[FactorProgram, Node, list[str], str] | None:
        for _ in range(self.cfg.max_resample_tries):
            tree, pids, actual = self._apply_operator(op_name, island, max_depth)
            program = self._render(tree)
            if self._is_duplicate(program) or not self._acceptable(program):
                continue
            return program, tree, pids, actual
        return None

    def _apply_operator(self, op_name: str, island: int, max_depth: int,
                        ) -> tuple[Node, list[str], str]:
        """Apply one GP operator, returning (child_tree, parent_ids, op_used)."""
        if op_name == "crossover":
            parents = self.controller.select_parents(2, island=island)
            trees = [t for t in (self._tree_for(p) for p in parents) if t is not None]
            if len(trees) >= 2:
                child = operators.subtree_crossover(
                    trees[0], trees[1], self.grammar, self.rng, max_depth)
                pids = [p.genome.program.factor_id for p in parents[:2]]
                return child, pids, "crossover"
            op_name = "subtree"  # not enough parents → fall back to a unary op

        parents = self.controller.select_parents(1, island=island)
        tree = self._tree_for(parents[0]) if parents else None
        if tree is None:
            return random_tree(self.grammar, self.rng, max_depth), [], "seed"
        pids = [parents[0].genome.program.factor_id]
        if op_name == "point":
            return operators.point_mutation(tree, self.grammar, self.rng), pids, "point"
        if op_name == "hoist":
            return operators.hoist_mutation(tree, self.rng), pids, "hoist"
        # default: subtree mutation
        return (operators.subtree_mutation(tree, self.grammar, self.rng, max_depth),
                pids, "subtree")

    def _render(self, tree: Node) -> FactorProgram:
        fid = self._fresh_id()
        return tree_to_program(tree, fid, self.grammar,
                               horizon=self.cfg.target_horizon,
                               category=self.cfg.category)

    # ── evaluation (kept in sync with EvolutionLoop.evaluate_program) ──

    def evaluate_program(self, program: FactorProgram, *,
                         include_probes: bool = True,
                         include_reference: bool = True,
                         book_override: list[tuple[str, str]] | None = None,
                         bill: bool = True) -> FitnessResult | None:
        """Score one program via the same MCP/in-process seam the LLM arm uses.

        Under progressive reveal the split is calendar-mode (``is_end`` /
        ``val_end`` from the current window) instead of fractional.  The
        rescore path passes ``include_probes=False`` (plateau dock carried
        forward instead), a snapshot ``book_override`` for order-independence
        and ``bill=False`` (re-scores are not trials).
        """
        from quant_fund_agent.agents.factor_research.evolution.mutation import (
            jitter_variants,
        )
        from quant_fund_agent.mcp import research_client

        book = [
            {"factor_id": b["factor_id"], "code": b["code"]}
            for b in self.fixed_book
            if b["factor_id"] != program.factor_id
        ]
        fixed_ids = {b["factor_id"] for b in book}
        loco = (book_override if book_override is not None
                else list(self.controller.archive_programs()))
        book.extend(
            {"factor_id": fid, "code": code}
            for fid, code in loco
            if fid != program.factor_id and fid not in fixed_ids
        )
        probes = ([{"factor_id": pid, "code": pcode}
                   for pid, pcode in jitter_variants(program, self.cfg.plateau_scales)]
                  if include_probes else [])
        reference = ([{"factor_id": r["factor_id"], "code": r["code"]}
                      for r in self.reference_book
                      if r.get("factor_id") != program.factor_id]
                     if include_reference else [])

        split: dict[str, Any] = {"is_frac": self.cfg.is_frac,
                                 "val_frac": self.cfg.val_frac}
        if self._window is not None:
            split = {"is_end": self._window.is_end_ts,
                     "val_end": self._window.val_end_ts}
        res = research_client.evaluate_fitness(
            candidate={"factor_id": program.factor_id, "code": program.code,
                       "expected_sign": program.expected_sign},
            book=book,
            jitter=probes,
            reference=reference or None,
            target_horizon=self.cfg.target_horizon,
            n_trials=self.controller.n_trials + (1 if bill else 0),
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
            **split,
        )
        if not res.get("ok"):
            log.info("[%s] evaluation failed: %s", program.factor_id, res.get("error"))
            self.failures.append({"factor_id": program.factor_id,
                                  "error": res.get("error")})
            return None
        if bill:
            self.controller.next_trial()
        fitness = FitnessResult.from_dict(res["fitness"])
        if self.cfg.level_rho_max is not None:
            rho = (fitness.diagnostics or {}).get("level_rho")
            if rho is not None and rho > self.cfg.level_rho_max:
                log.info("[%s] level gate: rho %.4f > %.3f — rejected",
                         program.factor_id, rho, self.cfg.level_rho_max)
                self.failures.append({"factor_id": program.factor_id,
                                      "error": f"level_rho {rho:.4f}"})
                return None
        return fitness

    # ── admission ──

    def _admit(self, program: FactorProgram, tree: Node, *, generation: int,
               island: int, operator: str,
               parent_ids: Sequence[str]) -> EvaluatedGenome | None:
        genome = Genome(
            genome_id=f"g{generation}-{program.factor_id}-{uuid.uuid4().hex[:6]}",
            programs=[program], unit="single", generation=generation,
            island=island, parent_ids=list(parent_ids), operator=operator,
            metadata={"gp_tree": tree.to_dict()},
        )
        if self.controller.is_duplicate(genome):
            return None
        fitness = self.evaluate_program(program)
        if fitness is None:
            return None
        eg = EvaluatedGenome(genome=genome, fitness=fitness)
        self.controller.insert(eg)
        self._trees[program.factor_id] = tree
        return eg

    # ── progressive reveal (faithful port of the LLM arm's machinery) ──

    def _total_generations(self) -> int:
        return (len(tuple(self.cfg.depth_schedule) or (5,))
                * max(1, self.cfg.generations))

    def _init_progressive(self) -> None:
        """Build the per-generation reveal schedule over the panel timeline."""
        import pandas as pd
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
            index, generations=self._total_generations(),
            test_frac=self.cfg.test_frac, seed_frac=self.cfg.seed_frac,
            reveal_every=self.cfg.reveal_every, val_blocks=self.cfg.val_blocks,
            holdout_last=False,
            wf_blocks=self.cfg.wf_blocks, wf_block_bars=self.cfg.wf_block_bars)
        g = min(self.controller.generation, self._total_generations())
        self._window = self._schedule[g]
        log.info("progressive reveal (gp): %d windows; generation %d frontier "
                 "through %s", len(self._schedule), self.controller.generation,
                 self._window.val_end_ts)

    def _reveal_index(self, gen: int) -> int | None:
        gens = [w.generation for w in self._schedule if w.reveal]
        return gens.index(gen) if gen in gens else None

    def _prequential_score(self, gen: int, old_ts: str, new_ts: str) -> None:
        """Honest OOS score of the archive on the block about to be revealed."""
        from quant_fund_agent.mcp import research_client

        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "prequential.jsonl"
        if self._prequential_done is None:
            done: set[int] = set()
            if path.exists():
                for line in path.read_text().splitlines():
                    try:
                        done.add(int(json.loads(line).get("generation")))
                    except (ValueError, TypeError):
                        pass
            self._prequential_done = done
        if gen in self._prequential_done:
            return
        book = self.controller.archive_programs()
        if not book:
            row: dict[str, Any] = {"generation": gen, "start": old_ts,
                                   "end": new_ts, "skipped": "empty archive"}
        else:
            res = research_client.score_book_oos(
                [{"factor_id": fid, "code": code} for fid, code in book],
                start=old_ts, end=new_ts,
                target_horizon=self.cfg.target_horizon,
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
        row["reveal_index"] = self._reveal_index(gen)
        row["frontier_ts"] = str(new_ts)
        if gen < len(self._schedule):
            row["visible_end"] = self._schedule[gen].visible_end
        with path.open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        self._prequential_done.add(gen)

    def _rescore_archive_on_window(self, gen: int) -> None:
        """Re-score every archive member on the advanced window (no billing)."""
        archive = list(self.controller.archive)
        snapshot = list(self.controller.archive_programs())
        new_fitness: dict[str, FitnessResult] = {}
        for eg in archive:
            obj_before = eg.fitness.objective.to_dict()
            fitness = self.evaluate_program(
                eg.genome.program, include_probes=False,
                include_reference=False, book_override=snapshot, bill=False)
            if fitness is None:
                self.controller.lineage.append({
                    "event": "rescore_failed", "generation": gen,
                    "genome_id": eg.genome.genome_id})
                continue
            # carry the admission-time plateau dock forward (probes skipped here)
            prev = eg.fitness.diagnostics or {}
            carried = prev.get("plateau_penalty")
            if carried is None:
                carried = prev.get("plateau_penalty_carried")
            if carried and fitness.objective.marginal_value is not None:
                fitness.objective.marginal_value = (
                    float(fitness.objective.marginal_value) - float(carried))
                fitness.diagnostics["plateau_penalty_carried"] = float(carried)
            new_fitness[eg.genome.genome_id] = fitness
            self.controller.lineage.append({
                "event": "rescore", "generation": gen,
                "genome_id": eg.genome.genome_id,
                "objective_before": obj_before,
                "objective_after": fitness.objective.to_dict(),
                "gates_after": fitness.gates.to_dict(),
                "reveal_index": self._reveal_index(gen),
            })
        self.controller.rescore_archive(new_fitness)

    def _advance_reveal(self, gen: int) -> None:
        """Prequential probe → advance frontier → rescore → free gate-failers."""
        new_window = self._schedule[gen]
        self._prequential_score(gen, self._window.val_end_ts,
                                new_window.val_end_ts)
        self._window = new_window
        self._rescore_archive_on_window(gen)
        released = self.controller.release_failed_fingerprints()
        if released:
            log.info("generation %d: released %d gate-failing fingerprint(s)",
                     gen, released)

    # ── checkpoint (same three files as the LLM arm) ──

    def _checkpoint(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.controller.save(self.out_dir / "state.json")
        (self.out_dir / "run_config.json").write_text(
            json.dumps(self.cfg.to_dict(), indent=2))
        with (self.out_dir / "lineage.jsonl").open("w") as fh:
            for row in self.controller.lineage:
                fh.write(json.dumps(row, default=str) + "\n")
        if self._window is not None:
            (self.out_dir / "progressive.json").write_text(json.dumps({
                "frontier_ts": self._window.val_end_ts,
                "generation": self.controller.generation}))

    # ── main drive ──

    def run(self, initial_trees: Sequence[Node] | None = None) -> dict[str, Any]:
        """Run the whole GP search; returns a summary dict.

        ``initial_trees`` (mostly for tests / resumed runs) bypasses random
        seeding and admits exactly those trees at generation 0.
        """
        cfg = self.cfg
        t0 = time.time()
        self._load_known_ids()
        n_islands = max(1, cfg.n_islands)
        stages = tuple(cfg.depth_schedule) or (5,)
        if cfg.progressive:
            self._init_progressive()

        # ── generation 0: seed ──
        seed_depth = stages[0]
        if initial_trees is not None:
            for k, tree in enumerate(initial_trees):
                program = self._render(tree)
                if self._is_duplicate(program) or not self._acceptable(program):
                    continue
                self._admit(program, tree, generation=0, island=k % n_islands,
                            operator="gp_seed", parent_ids=[])
        else:
            for k in range(cfg.seed_pop):
                method = "full" if k % 2 == 0 else "grow"
                made = self._seed_one(seed_depth, method)
                if made is None:
                    continue
                program, tree = made
                self._admit(program, tree, generation=0, island=k % n_islands,
                            operator="gp_seed", parent_ids=[])
        self._checkpoint()
        log.info("generation 0: population=%d, archive=%d, n_trials=%d",
                 len(self.controller.population()), len(self.controller.archive),
                 self.controller.n_trials)
        if not self.controller.population():
            log.warning("empty seed population — aborting run")
            return self.summary(time.time() - t0)

        op_names, op_probs = self._operator_mix()

        global_gen = 0
        for max_depth in stages:
            for _ in range(cfg.generations):
                global_gen += 1
                self.controller.generation = global_gen
                # progressive reveal: prequential probe, frontier advance and
                # archive rescore all happen BEFORE any child of this
                # generation is proposed (identical order to the LLM arm).
                if cfg.progressive and self._schedule[global_gen].reveal:
                    self._advance_reveal(global_gen)
                for c in range(cfg.children_per_generation):
                    island = c % n_islands
                    op_name = str(self.rng.choice(op_names, p=op_probs))
                    proposed = self._propose(op_name, island, max_depth)
                    if proposed is None:
                        continue
                    program, tree, pids, actual = proposed
                    self._admit(program, tree, generation=global_gen, island=island,
                                operator=f"gp_{actual}", parent_ids=pids)
                if global_gen % max(1, cfg.migration_every) == 0:
                    self.controller.migrate()
                self._checkpoint()
                log.info("stage_depth=%d gen=%d: population=%d, archive=%d, "
                         "n_trials=%d", max_depth, global_gen,
                         len(self.controller.population()),
                         len(self.controller.archive), self.controller.n_trials)

        return self.summary(time.time() - t0)

    def _operator_mix(self) -> tuple[list[str], list[float]]:
        raw = [
            ("crossover", max(0.0, self.cfg.p_crossover)),
            ("subtree", max(0.0, self.cfg.p_subtree)),
            ("point", max(0.0, self.cfg.p_point)),
            ("hoist", max(0.0, self.cfg.p_hoist)),
        ]
        total = sum(w for _, w in raw) or 1.0
        names = [n for n, _ in raw]
        probs = [w / total for _, w in raw]
        return names, probs

    def summary(self, elapsed: float) -> dict[str, Any]:
        return {
            "engine": "gp",
            "generations": self.controller.generation,
            "n_trials": self.controller.n_trials,
            "population": len(self.controller.population()),
            "archive": [
                {"factor_ids": eg.genome.factor_ids,
                 "genome_id": eg.genome.genome_id,
                 "operator": eg.genome.operator,
                 "generation": eg.genome.generation,
                 "objective": eg.fitness.objective.to_dict()}
                for eg in self.controller.archive
            ],
            "n_eval_failures": len(self.failures),
            "elapsed_sec": round(elapsed, 1),
        }
