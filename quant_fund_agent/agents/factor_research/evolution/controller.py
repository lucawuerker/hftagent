"""Deterministic evolution controller: NSGA-II selection, archive, islands, N_trials.

This is the *selection* half of the loop — it only ever consumes the deterministic
:class:`~quant_fund_agent.research_eval.fitness.FitnessResult` (objective vector +
gate booleans) and never a word of LLM output, so the LLM cannot influence its own
reward (the core design principle).

Selection is **constrained NSGA-II** (Deb 2002):

* *Constrained dominance* — a gate-passing candidate always dominates a
  gate-failing one; two gate-failers compare by how many gates they fail; only
  gate-passing pairs (or equally-failing pairs) compare on the Pareto objective
  vector.  This is exactly the "hard gates, else treated as dominated" rule from
  the design, made total enough to rank a population where nobody passes yet.
* *Non-dominated sort* into fronts + *crowding distance* within a front, so
  within-front selection pressure favours diverse candidates.
* *Binary tournament* on (front rank, crowding) picks parents.

The **archive** is the gate-passing non-dominated front across everything ever
evaluated — per the locked decision it *is* the "accepted book" that SINGLE-mode
marginal value is scored against, and it is what gets persisted at the end.

The controller also owns the ``N_trials`` counter (every evaluated candidate
increments it, feeding the deflation gate — the search is a multiple-testing
machine and must be billed for every look) and the lineage log for the thesis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from quant_fund_agent.agents.factor_research.evolution.genome import Genome
from quant_fund_agent.research_eval.fitness import FitnessResult

log = logging.getLogger("evolution.controller")


@dataclass
class EvaluatedGenome:
    """A genome paired with its deterministic fitness (the population element)."""

    genome: Genome
    fitness: FitnessResult

    @property
    def selectable(self) -> bool:
        return self.fitness.selectable

    def to_dict(self) -> dict[str, Any]:
        return {"genome": self.genome.to_dict(), "fitness": self.fitness.to_dict()}


# ── constrained dominance / sorting primitives ────────────────────────────────

def _n_failed_gates(f: FitnessResult) -> int:
    return sum(1 for g in f.gates.GATES if getattr(f.gates, g) is False)


def constrained_dominates(a: FitnessResult, b: FitnessResult) -> bool:
    """Deb's constrained-dominance: feasibility first, then Pareto dominance.

    ``a`` dominates ``b`` when (i) ``a`` passes the gates and ``b`` doesn't;
    (ii) both fail but ``a`` fails strictly fewer gates; or (iii) both pass (or
    fail equally many) and ``a`` Pareto-dominates ``b`` on the objective vector.
    """
    from quant_fund_agent.research_eval.fitness import dominates

    a_ok, b_ok = a.gates.passed, b.gates.passed
    if a_ok and not b_ok:
        return True
    if b_ok and not a_ok:
        return False
    if not a_ok and not b_ok:
        fa, fb = _n_failed_gates(a), _n_failed_gates(b)
        if fa != fb:
            return fa < fb
    return dominates(a.objective, b.objective)


def non_dominated_sort(results: Sequence[FitnessResult]) -> list[list[int]]:
    """Fast non-dominated sort under constrained dominance → fronts of indices.

    Front 0 is the (constrained) Pareto front; every candidate appears in exactly
    one front.  O(n²) in the population size, which is tiny here (tens).
    """
    n = len(results)
    dominated_by: list[list[int]] = [[] for _ in range(n)]  # i dominates these
    dom_count = [0] * n                                      # how many dominate i

    for i in range(n):
        for j in range(i + 1, n):
            if constrained_dominates(results[i], results[j]):
                dominated_by[i].append(j)
                dom_count[j] += 1
            elif constrained_dominates(results[j], results[i]):
                dominated_by[j].append(i)
                dom_count[i] += 1

    fronts: list[list[int]] = []
    current = [i for i in range(n) if dom_count[i] == 0]
    while current:
        fronts.append(current)
        nxt: list[int] = []
        for i in current:
            for j in dominated_by[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    nxt.append(j)
        current = nxt
    return fronts


def crowding_distance(results: Sequence[FitnessResult],
                      front: Sequence[int]) -> dict[int, float]:
    """NSGA-II crowding distance of each index in ``front`` (bigger = lonelier).

    Boundary candidates on any axis get ``inf`` so the extremes of the front are
    always kept — that is what maintains spread along the Pareto surface.
    ``None`` axes enter as ``-inf`` via ``ObjectiveVector.as_tuple`` and simply
    tie at the bottom.
    """
    dist = {i: 0.0 for i in front}
    if len(front) <= 2:
        return {i: float("inf") for i in front}

    n_axes = len(results[front[0]].objective.as_tuple())
    for ax in range(n_axes):
        vals = [(results[i].objective.as_tuple()[ax], i) for i in front]
        vals.sort()
        lo, hi = vals[0][0], vals[-1][0]
        dist[vals[0][1]] = float("inf")
        dist[vals[-1][1]] = float("inf")
        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            continue
        for k in range(1, len(vals) - 1):
            prev_v, next_v = vals[k - 1][0], vals[k + 1][0]
            if np.isfinite(prev_v) and np.isfinite(next_v):
                dist[vals[k][1]] += (next_v - prev_v) / span
    return dist


def _rank_population(pop: Sequence[EvaluatedGenome]) -> list[tuple[int, float, int]]:
    """(front_rank, -crowding, index) sort keys for a population, best first."""
    results = [eg.fitness for eg in pop]
    fronts = non_dominated_sort(results)
    keyed: list[tuple[int, float, int]] = []
    for rank, front in enumerate(fronts):
        crowd = crowding_distance(results, front)
        for i in front:
            keyed.append((rank, -crowd[i], i))
    keyed.sort()
    return keyed


# ── the controller ────────────────────────────────────────────────────────────

@dataclass
class ControllerConfig:
    population_size: int = 12          # per island
    n_islands: int = 1
    migration_every: int = 5           # generations between elite migrations
    migration_k: int = 1               # elites moved per island per migration
    seed: int = 0
    # ── QD selection (WS2) ──
    # "nsga2" (default): the Pareto archive drives parent selection (existing arm).
    # "qd": a MAP-Elites behavior grid drives parent selection + the accepted book,
    # keeping the SAME 5-axis Pareto as the per-cell quality (a clean ablation arm).
    selection: str = "nsga2"
    grid_dims: int = 2                 # QD behavior-grid dimensionality (2 or 3)
    cell_capacity: int = 3             # QD mini-Pareto elites per cell
    depth_gamma: float = 0.0           # P7 depth penalty on QD parent sampling (0 = off)
    reuse_omega: float = 0.0           # P7 parent-reuse penalty on QD parent sampling


class EvolutionController:
    """Population + islands + archive + N_trials + lineage, all deterministic."""

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config or ControllerConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.islands: list[list[EvaluatedGenome]] = [
            [] for _ in range(max(1, self.config.n_islands))]
        self.archive: list[EvaluatedGenome] = []
        # Two-stage curation (Lever 2): every gate-passing genome ever evaluated,
        # regardless of whether it survives the Pareto archive.  When curation is
        # enabled this pool — not the (domination-pruned) archive — is what gets
        # curated into the final book, so a factor is never discarded merely for
        # being dominated.  Always populated; only *used* when curation != archive.
        self.kept_pool: list[EvaluatedGenome] = []
        self._pool_fingerprints: set[str] = set()
        self.n_trials: int = 0
        self.generation: int = 0
        self.lineage: list[dict[str, Any]] = []
        self._fingerprints: set[str] = set()
        # ── QD selection (WS2): a behavior grid alongside the Pareto archive ──
        self.qd = None
        if self.config.selection == "qd":
            from quant_fund_agent.agents.factor_research.evolution.qd import (
                QDArchive,
                QDConfig,
            )
            self.qd = QDArchive(QDConfig(
                grid_dims=self.config.grid_dims,
                cell_capacity=self.config.cell_capacity,
                depth_gamma=self.config.depth_gamma,
                reuse_omega=self.config.reuse_omega,
            ))
        # P7: how many times each genome has been drawn as a parent (QD sampling bias)
        self.parent_reuse: dict[str, int] = {}

    # ── trials / dedup ──

    def next_trial(self) -> int:
        """Bill one evaluation against the multiple-testing budget → new N_trials.

        Called *before* the harness scores a candidate so the candidate's own
        deflation gate already counts itself among the trials.
        """
        self.n_trials += 1
        return self.n_trials

    def is_duplicate(self, genome: Genome) -> bool:
        """True if a byte-equivalent program set was already evaluated.

        Duplicates are rejected *before* evaluation so they don't inflate
        ``N_trials`` (they carry no new information) and don't crowd the
        population with clones.
        """
        return genome.code_fingerprint() in self._fingerprints

    # ── insertion ──

    def insert(self, evaluated: EvaluatedGenome) -> None:
        """Add an evaluated genome to its island, truncate, update the archive."""
        genome = evaluated.genome
        self._fingerprints.add(genome.code_fingerprint())

        island_idx = genome.island % len(self.islands)
        island = self.islands[island_idx]
        island.append(evaluated)
        if len(island) > self.config.population_size:
            keep = _rank_population(island)[: self.config.population_size]
            self.islands[island_idx] = [island[i] for _, _, i in keep]

        self._update_archive(evaluated)
        if self.qd is not None:
            self.qd.insert(evaluated)   # QD behavior grid (only gate-passers occupy cells)
        if evaluated.selectable:
            fp = genome.code_fingerprint()
            if fp not in self._pool_fingerprints:
                self._pool_fingerprints.add(fp)
                self.kept_pool.append(evaluated)
        self.lineage.append({
            "genome_id": genome.genome_id,
            "factor_ids": genome.factor_ids,
            "generation": genome.generation,
            "island": island_idx,
            "operator": genome.operator,
            "parent_ids": genome.parent_ids,
            "n_trials_at_eval": evaluated.fitness.diagnostics.get("n_trials"),
            "archive_size_at_eval": len(self.archive),
            "objective": evaluated.fitness.objective.to_dict(),
            "gates": evaluated.fitness.gates.to_dict(),
            "selectable": evaluated.selectable,
        })

    def _update_archive(self, evaluated: EvaluatedGenome) -> None:
        """Archive = gate-passing non-dominated set over everything evaluated.

        Gate-failers never enter (the archive is the *accepted book* that
        marginal value is scored against — it must stay clean even when the
        whole population is failing early in a run; selection then still works
        through the constrained sort's feasibility ordering).
        """
        if not evaluated.selectable:
            return
        from quant_fund_agent.research_eval.fitness import dominates

        pool = self.archive + [evaluated]
        new_archive: list[EvaluatedGenome] = []
        for eg in pool:
            if not any(dominates(o.fitness.objective, eg.fitness.objective)
                       for o in pool if o is not eg):
                new_archive.append(eg)
        self.archive = new_archive

    # ── selection ──

    def population(self, island: int | None = None) -> list[EvaluatedGenome]:
        if island is not None:
            return list(self.islands[island % len(self.islands)])
        return [eg for isl in self.islands for eg in isl]

    def select_parents(self, k: int, island: int | None = None) -> list[EvaluatedGenome]:
        """``k`` parents by binary tournament on (front rank, crowding distance).

        Draws from one island (or the whole population) with replacement across
        tournaments — the classic NSGA-II parent-selection pressure: mostly the
        front, but diverse and occasionally an underdog.
        """
        # ── QD parent selection: sample an occupied cell → one elite (P7-biased) ──
        if self.qd is not None and self.qd.n_cells() > 0:
            parents: list[EvaluatedGenome] = []
            for _ in range(k):
                eg = self.qd.sample_parent(self.rng, reuse_counts=self.parent_reuse)
                if eg is None:
                    break
                self.parent_reuse[eg.genome.genome_id] = \
                    self.parent_reuse.get(eg.genome.genome_id, 0) + 1
                parents.append(eg)
            if parents:
                return parents
            # QD grid still empty (gen 0 before any gate-passer) → fall through to NSGA

        pop = self.population(island)
        if not pop:
            return []
        keyed = _rank_population(pop)
        rank_of = {i: (rank, neg_crowd) for rank, neg_crowd, i in keyed}

        parents = []
        for _ in range(k):
            i, j = self.rng.integers(0, len(pop), size=2)
            winner = i if rank_of[int(i)] <= rank_of[int(j)] else j
            parents.append(pop[int(winner)])
        return parents

    def migrate(self) -> int:
        """Copy each island's top ``migration_k`` elites to the next island (ring).

        Returns the number of genomes moved.  A no-op with one island.
        """
        n_isl = len(self.islands)
        if n_isl < 2:
            return 0
        moved = 0
        elites_per_island: list[list[EvaluatedGenome]] = []
        for island in self.islands:
            if not island:
                elites_per_island.append([])
                continue
            keyed = _rank_population(island)[: self.config.migration_k]
            elites_per_island.append([island[i] for _, _, i in keyed])
        for src, elites in enumerate(elites_per_island):
            dst = (src + 1) % n_isl
            for eg in elites:
                if any(o.genome.genome_id == eg.genome.genome_id
                       for o in self.islands[dst]):
                    continue
                self.islands[dst].append(eg)
                moved += 1
            if len(self.islands[dst]) > self.config.population_size:
                keep = _rank_population(self.islands[dst])[: self.config.population_size]
                self.islands[dst] = [self.islands[dst][i] for _, _, i in keep]
        return moved

    # ── the book the harness scores against ──

    def accepted_book(self) -> list[EvaluatedGenome]:
        """The genomes that form the accepted book — the QD cell elites in QD mode,
        else the Pareto archive.  This is the SINGLE marginal-value reference and the
        default persist source, so the two selection modes share one seam."""
        if self.qd is not None:
            return self.qd.elites()
        return self.archive

    def archive_programs(self) -> list[tuple[str, str]]:
        """The accepted book as ``(factor_id, code)`` pairs (SINGLE marginal ref).

        De-duplicated by factor id — in SET mode archive genomes may share
        members; the book is the union of member programs.  In QD mode the book is
        the union of the behavior-grid cell elites (the diverse archive), keeping the
        non-stationary LOCO marginal-value semantics vs a *diverse* reference.
        """
        seen: dict[str, str] = {}
        for eg in self.accepted_book():
            for p in eg.genome.programs:
                seen.setdefault(p.factor_id, p.code)
        return list(seen.items())

    def kept_pool_programs(self) -> list[tuple[str, str]]:
        """Every gate-passing factor ever evaluated as ``(factor_id, code)`` pairs.

        The two-stage curation input (Lever 2): de-duplicated by factor id across
        all kept genomes (SET genomes may share members).  Falls back to the
        archive when the pool is empty (e.g. a legacy state file).
        """
        seen: dict[str, str] = {}
        for eg in (self.kept_pool or self.archive):
            for p in eg.genome.programs:
                seen.setdefault(p.factor_id, p.code)
        return list(seen.items())

    # ── persistence (resumability + thesis audit) ──

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "population_size": self.config.population_size,
                "n_islands": self.config.n_islands,
                "migration_every": self.config.migration_every,
                "migration_k": self.config.migration_k,
                "seed": self.config.seed,
                "selection": self.config.selection,
                "grid_dims": self.config.grid_dims,
                "cell_capacity": self.config.cell_capacity,
                "depth_gamma": self.config.depth_gamma,
                "reuse_omega": self.config.reuse_omega,
            },
            "n_trials": self.n_trials,
            "generation": self.generation,
            "islands": [[eg.to_dict() for eg in isl] for isl in self.islands],
            "archive": [eg.to_dict() for eg in self.archive],
            "kept_pool": [eg.to_dict() for eg in self.kept_pool],
            "fingerprints": sorted(self._fingerprints),
            "parent_reuse": dict(self.parent_reuse),
            "qd": self.qd.to_dict() if self.qd is not None else None,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state_dict(), indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "EvolutionController":
        payload = json.loads(Path(path).read_text())
        cfgd = payload.get("config", {})
        ctrl = cls(ControllerConfig(
            population_size=cfgd.get("population_size", 12),
            n_islands=cfgd.get("n_islands", 1),
            migration_every=cfgd.get("migration_every", 5),
            migration_k=cfgd.get("migration_k", 1),
            seed=cfgd.get("seed", 0),
            selection=cfgd.get("selection", "nsga2"),
            grid_dims=cfgd.get("grid_dims", 2),
            cell_capacity=cfgd.get("cell_capacity", 3),
            depth_gamma=cfgd.get("depth_gamma", 0.0),
            reuse_omega=cfgd.get("reuse_omega", 0.0),
        ))

        def _eg(d: dict[str, Any]) -> EvaluatedGenome:
            return EvaluatedGenome(genome=Genome.from_dict(d["genome"]),
                                   fitness=FitnessResult.from_dict(d["fitness"]))

        ctrl.n_trials = payload.get("n_trials", 0)
        ctrl.generation = payload.get("generation", 0)
        ctrl.islands = [[_eg(d) for d in isl] for isl in payload.get("islands", [[]])]
        if not ctrl.islands:
            ctrl.islands = [[]]
        ctrl.archive = [_eg(d) for d in payload.get("archive", [])]
        ctrl.kept_pool = [_eg(d) for d in payload.get("kept_pool", [])]
        ctrl._pool_fingerprints = {
            eg.genome.code_fingerprint() for eg in ctrl.kept_pool}
        ctrl._fingerprints = set(payload.get("fingerprints", []))
        ctrl.parent_reuse = dict(payload.get("parent_reuse", {}))
        qd_payload = payload.get("qd")
        if qd_payload is not None:
            from quant_fund_agent.agents.factor_research.evolution.qd import QDArchive
            ctrl.qd = QDArchive.from_dict(qd_payload)
        return ctrl
