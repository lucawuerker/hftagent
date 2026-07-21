"""Deterministic evolution controller: grouped NSGA-II demes and N_trials.

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

Every knowledge-graph mechanism group owns a gate-passing non-dominated archive.
The accepted book is their union: no mechanism group can be erased by a globally
stronger group before final curation.

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
    n_mechanism_groups: int = 1
    demes_per_group: int = 1
    # Legacy alias.  Old checkpoints/callers used ``n_islands`` for the single
    # layer; when provided it means demes per group.
    n_islands: int | None = None
    migration_every: int = 5           # generations between elite migrations
    migration_k: int = 1               # elites moved per island per migration
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_islands is not None:
            self.demes_per_group = max(1, int(self.n_islands))
        self.n_mechanism_groups = max(1, int(self.n_mechanism_groups))
        self.demes_per_group = max(1, int(self.demes_per_group))


class EvolutionController:
    """Population + islands + archive + N_trials + lineage, all deterministic."""

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config or ControllerConfig()
        self.rng = np.random.default_rng(self.config.seed)
        n_demes = self.config.n_mechanism_groups * self.config.demes_per_group
        self.islands: list[list[EvaluatedGenome]] = [
            [] for _ in range(n_demes)]
        self.group_archives: list[list[EvaluatedGenome]] = [
            [] for _ in range(self.config.n_mechanism_groups)]
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
        # Progressive reveal: fingerprints of gate-FAILING genomes and how many times
        # each has been released for a retry.  On newly-revealed data a genome that
        # failed a gate before may honestly pass, so its fingerprint is freed once (cap
        # = 1 retry) so an identical genome isn't re-tried every reveal forever.
        self._failed_fingerprints: dict[str, int] = {}
        self.parent_reuse: dict[str, int] = {}

    @property
    def archive(self) -> list[EvaluatedGenome]:
        """Union of the reserved per-mechanism archives (legacy public seam)."""
        return [eg for group in self.group_archives for eg in group]

    @archive.setter
    def archive(self, values: Sequence[EvaluatedGenome]) -> None:
        self.group_archives = [[] for _ in range(self.config.n_mechanism_groups)]
        for eg in values:
            group = eg.genome.mechanism_group_id % len(self.group_archives)
            self.group_archives[group].append(eg)

    def flat_island(self, mechanism_group_id: int, deme_id: int) -> int:
        group = mechanism_group_id % self.config.n_mechanism_groups
        deme = deme_id % self.config.demes_per_group
        return group * self.config.demes_per_group + deme

    def coordinates(self, island: int) -> tuple[int, int]:
        flat = island % len(self.islands)
        return divmod(flat, self.config.demes_per_group)

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
        key = self._fingerprint_key(genome)
        return key in self._fingerprints

    def _fingerprint_key(self, genome: Genome) -> str:
        return (f"{genome.mechanism_group_id}:{genome.deme_id}:"
                f"{genome.code_fingerprint()}")

    # ── insertion ──

    def insert(self, evaluated: EvaluatedGenome) -> None:
        """Add an evaluated genome to its island, truncate, update the archive."""
        genome = evaluated.genome
        self._fingerprints.add(self._fingerprint_key(genome))

        island_idx = self.flat_island(
            genome.mechanism_group_id, genome.deme_id)
        genome.island = island_idx
        island = self.islands[island_idx]
        island.append(evaluated)
        if len(island) > self.config.population_size:
            keep = _rank_population(island)[: self.config.population_size]
            self.islands[island_idx] = [island[i] for _, _, i in keep]

        if not evaluated.selectable:
            # track gate-failers so progressive reveal can retry them on new data
            self._failed_fingerprints.setdefault(self._fingerprint_key(genome), 0)

        self._update_archive(evaluated)
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
            "mechanism_group_id": genome.mechanism_group_id,
            "deme_id": genome.deme_id,
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

        group_id = evaluated.genome.mechanism_group_id % len(self.group_archives)
        pool = self.group_archives[group_id] + [evaluated]
        new_archive: list[EvaluatedGenome] = []
        for eg in pool:
            if not any(dominates(o.fitness.objective, eg.fitness.objective)
                       for o in pool if o is not eg):
                new_archive.append(eg)
        self.group_archives[group_id] = new_archive

    # ── progressive reveal: retry gate-failers + re-score the archive ──

    def release_failed_fingerprints(self, max_retries: int = 1) -> int:
        """Free gate-failing fingerprints so identical genomes can be re-evaluated.

        On newly-revealed data a previously gate-failing genome may honestly pass, so
        its fingerprint is discarded from the dedup set (making it non-duplicate
        again).  Each fingerprint is released at most ``max_retries`` times so the same
        genome cannot be re-tried on every reveal forever.  Returns how many were
        released.
        """
        released = 0
        for fp, count in list(self._failed_fingerprints.items()):
            if count < max_retries:
                self._fingerprints.discard(fp)
                self._failed_fingerprints[fp] = count + 1
                released += 1
        return released

    def rescore_archive(
        self, new_fitness_by_genome_id: dict[str, FitnessResult],
    ) -> None:
        """Replace archive members' fitnesses with freshly-computed ones and re-prune.

        Progressive reveal calls this after the frontier advances: every archive
        member is re-scored on the new (larger) window, its fitness replaced in place,
        and the archive rebuilt as the gate-passing non-dominated set of the re-scored
        members.  Members that drop out stay in ``kept_pool`` (end-of-run curation
        refits on the final window anyway).  Deliberately does **not** touch
        ``n_trials`` (re-scores are not new trials) or the QD grid (a diversity
        library, not the marginal reference).
        """
        from quant_fund_agent.research_eval.fitness import dominates

        for eg in self.archive:
            newf = new_fitness_by_genome_id.get(eg.genome.genome_id)
            if newf is not None:
                eg.fitness = newf
        for group_id, archive in enumerate(self.group_archives):
            passing = [eg for eg in archive if eg.selectable]
            self.group_archives[group_id] = [
                eg for eg in passing
                if not any(dominates(o.fitness.objective, eg.fitness.objective)
                           for o in passing if o is not eg)
            ]

    # ── selection ──

    def population(self, island: int | None = None, *,
                   mechanism_group_id: int | None = None,
                   deme_id: int | None = None) -> list[EvaluatedGenome]:
        if mechanism_group_id is not None:
            if deme_id is not None:
                island = self.flat_island(mechanism_group_id, deme_id)
            else:
                start = (mechanism_group_id % self.config.n_mechanism_groups) \
                    * self.config.demes_per_group
                return [eg for isl in self.islands[
                    start:start + self.config.demes_per_group] for eg in isl]
        if island is not None:
            return list(self.islands[island % len(self.islands)])
        return [eg for isl in self.islands for eg in isl]

    def select_parents(self, k: int, island: int | None = None, *,
                       mechanism_group_id: int | None = None,
                       deme_id: int | None = None) -> list[EvaluatedGenome]:
        """``k`` parents by binary tournament on (front rank, crowding distance).

        Draws from one island (or the whole population) with replacement across
        tournaments — the classic NSGA-II parent-selection pressure: mostly the
        front, but diverse and occasionally an underdog.
        """
        pop = self.population(island, mechanism_group_id=mechanism_group_id,
                              deme_id=deme_id)
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
        if self.config.demes_per_group < 2:
            return 0
        moved = 0
        # Ring migration is strictly inside a mechanism group.  Information crosses
        # groups only through the explicit synthesis operator in the outer loop.
        for group in range(self.config.n_mechanism_groups):
            base = group * self.config.demes_per_group
            local = self.islands[base:base + self.config.demes_per_group]
            elites_per_deme: list[list[EvaluatedGenome]] = []
            for deme in local:
                keyed = _rank_population(deme)[: self.config.migration_k] if deme else []
                elites_per_deme.append([deme[i] for _, _, i in keyed])
            for src, elites in enumerate(elites_per_deme):
                dst = base + ((src + 1) % self.config.demes_per_group)
                for eg in elites:
                    if any(o.genome.genome_id == eg.genome.genome_id
                           for o in self.islands[dst]):
                        continue
                    self.islands[dst].append(eg)
                    moved += 1
                if len(self.islands[dst]) > self.config.population_size:
                    keep = _rank_population(self.islands[dst])[:self.config.population_size]
                    self.islands[dst] = [self.islands[dst][i] for _, _, i in keep]
        return moved

    # ── the book the harness scores against ──

    def accepted_book(self) -> list[EvaluatedGenome]:
        """Union of reserved mechanism-group Pareto archives."""
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

    def group_archive(self, mechanism_group_id: int) -> list[EvaluatedGenome]:
        """Reserved Pareto archive for one knowledge-graph mechanism group."""
        return list(self.group_archives[
            mechanism_group_id % len(self.group_archives)])

    # ── persistence (resumability + thesis audit) ──

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "population_size": self.config.population_size,
                "n_mechanism_groups": self.config.n_mechanism_groups,
                "demes_per_group": self.config.demes_per_group,
                "migration_every": self.config.migration_every,
                "migration_k": self.config.migration_k,
                "seed": self.config.seed,
            },
            "n_trials": self.n_trials,
            "generation": self.generation,
            "islands": [[eg.to_dict() for eg in isl] for isl in self.islands],
            "group_archives": [
                [eg.to_dict() for eg in archive]
                for archive in self.group_archives
            ],
            # Kept for older readers.  New readers use ``group_archives``.
            "archive": [eg.to_dict() for eg in self.archive],
            "kept_pool": [eg.to_dict() for eg in self.kept_pool],
            "fingerprints": sorted(self._fingerprints),
            "failed_fingerprints": dict(self._failed_fingerprints),
            "parent_reuse": dict(self.parent_reuse),
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
            n_mechanism_groups=cfgd.get("n_mechanism_groups", 1),
            demes_per_group=cfgd.get(
                "demes_per_group", cfgd.get("n_islands", 1)),
            migration_every=cfgd.get("migration_every", 5),
            migration_k=cfgd.get("migration_k", 1),
            seed=cfgd.get("seed", 0),
        ))

        def _eg(d: dict[str, Any]) -> EvaluatedGenome:
            return EvaluatedGenome(genome=Genome.from_dict(d["genome"]),
                                   fitness=FitnessResult.from_dict(d["fitness"]))

        ctrl.n_trials = payload.get("n_trials", 0)
        ctrl.generation = payload.get("generation", 0)
        loaded_islands = [[_eg(d) for d in isl]
                          for isl in payload.get("islands", [[]])]
        expected = ctrl.config.n_mechanism_groups * ctrl.config.demes_per_group
        ctrl.islands = (loaded_islands + [[] for _ in range(expected)])[:expected]
        grouped = payload.get("group_archives")
        if grouped is not None:
            ctrl.group_archives = [
                [_eg(d) for d in archive] for archive in grouped]
            ctrl.group_archives = (
                ctrl.group_archives
                + [[] for _ in range(ctrl.config.n_mechanism_groups)]
            )[:ctrl.config.n_mechanism_groups]
        else:
            ctrl.archive = [_eg(d) for d in payload.get("archive", [])]
        ctrl.kept_pool = [_eg(d) for d in payload.get("kept_pool", [])]
        ctrl._pool_fingerprints = {
            eg.genome.code_fingerprint() for eg in ctrl.kept_pool}
        ctrl._fingerprints = set(payload.get("fingerprints", []))
        ctrl._failed_fingerprints = {
            k: int(v) for k, v in payload.get("failed_fingerprints", {}).items()}
        ctrl.parent_reuse = dict(payload.get("parent_reuse", {}))
        return ctrl
