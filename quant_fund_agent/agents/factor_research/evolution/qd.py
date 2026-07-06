"""Quality-Diversity archive (MAP-Elites) for diverse factor discovery (WS2).

The problem QD fixes: NSGA-II *converges* — the small Pareto archive that drives
parent selection collapses the search to a few lineages, so the discovered library
is narrow.  This ``QDArchive`` decouples *diversity of search* (a behavior grid) from
*quality* (the unchanged 5-axis Pareto).  It does **not** replace the fitness axes:
each grid cell keeps a small capped mini-Pareto over the SAME
:class:`~quant_fund_agent.research_eval.fitness.ObjectiveVector`, so every carefully
chosen criterion is still a hard scoring objective.  Behavior descriptors
(``trend_reversal``, ``signal_speed``, ``stress_activation``) — computed by the
harness, NOT scored — only decide which cell a genome lands in.

Design points (from the agreed spec):
* 2D grid by default (``trend_reversal × signal_speed``); ``grid_dims=3`` adds
  ``stress_activation``.
* Fixed, interpretable bin edges by default.  If adaptive edges are ever used they
  must be frozen once and serialized — this class stores its ``bin_edges`` and never
  rebins, so the archive is stable across a run and on resume.
* Each cell = a capped mini-Pareto (default 3) pruned by the EXISTING constrained
  dominance + crowding distance (no new comparison logic).
* Parent selection samples an occupied cell uniformly, then one elite from it (with
  an optional light P7 depth/parent-reuse bias).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # avoid a circular import at module load
    from quant_fund_agent.agents.factor_research.evolution.controller import EvaluatedGenome


@dataclass
class QDConfig:
    """Behavior-grid geometry.  Fixed edges are the default (deterministic)."""

    grid_dims: int = 2                 # 2 = trend×speed; 3 adds stress_activation
    cell_capacity: int = 3             # max elites (mini-Pareto) per cell
    # fixed interpretable bin edges (a value ≥ edge advances one bin):
    trend_edges: tuple[float, ...] = (-0.33, 0.33)   # fade | neutral | momentum
    speed_edges: tuple[float, ...] = (0.5, 1.0)      # slow | medium | fast
    stress_edges: tuple[float, ...] = (0.9, 1.3)     # dormant | neutral | activated
    # P7 parent-sampling bias (AlphaPROBE): (1-γ)^depth · (1-ω)^reuse.  0 = uniform.
    depth_gamma: float = 0.0
    reuse_omega: float = 0.0


def _bucket(value: Any, edges: tuple[float, ...]) -> int:
    """Bin index of ``value`` under fixed ``edges``.  Non-finite → bin 0 (fallback)."""
    if value is None:
        return 0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0
    if not np.isfinite(v):
        return 0
    b = 0
    for e in edges:
        if v >= e:
            b += 1
    return b


class QDArchive:
    """MAP-Elites grid of capped mini-Pareto cells keyed by binned descriptors."""

    def __init__(self, config: QDConfig | None = None) -> None:
        self.config = config or QDConfig()
        self.cells: dict[tuple[int, ...], list[EvaluatedGenome]] = {}
        # frozen bin edges — fixed here; the seam exists so an adaptive scheme could
        # compute them once (from gen-0) and store them, never rebinning mid-run.
        self.bin_edges: dict[str, tuple[float, ...]] = {
            "trend": tuple(self.config.trend_edges),
            "speed": tuple(self.config.speed_edges),
            "stress": tuple(self.config.stress_edges),
        }

    # ── cell keying ──
    def cell_key(self, behavior: dict[str, Any]) -> tuple[int, ...]:
        tr = _bucket(behavior.get("trend_reversal"), self.bin_edges["trend"])
        sp = _bucket(behavior.get("signal_speed"), self.bin_edges["speed"])
        if self.config.grid_dims >= 3:
            st = _bucket(behavior.get("stress_activation"), self.bin_edges["stress"])
            return (tr, sp, st)
        return (tr, sp)

    # ── insertion (cell competition → capped mini-Pareto) ──
    def insert(self, eg: "EvaluatedGenome") -> tuple[int, ...] | None:
        """Place a *gate-passing* genome in its cell; prune the cell to capacity.

        Returns the cell key it landed in, or ``None`` if it was not gate-passing
        (only ``search_selectable`` genomes occupy cells).
        """
        if not eg.selectable:
            return None
        key = self.cell_key(eg.fitness.behavior or {})
        cell = self.cells.setdefault(key, [])
        # dedup by genome id so a re-inserted elite doesn't clone
        if any(o.genome.genome_id == eg.genome.genome_id for o in cell):
            return key
        cell.append(eg)
        if len(cell) > self.config.cell_capacity:
            # keep the mini-Pareto best `cell_capacity` by the EXISTING NSGA-II ranking
            from quant_fund_agent.agents.factor_research.evolution.controller import (
                _rank_population,
            )
            keep = _rank_population(cell)[: self.config.cell_capacity]
            self.cells[key] = [cell[i] for _, _, i in keep]
        return key

    # ── parent sampling (uniform cell → elite, optional P7 bias) ──
    def sample_parent(self, rng: np.random.Generator, *,
                      reuse_counts: dict[str, int] | None = None) -> "EvaluatedGenome | None":
        keys = list(self.cells.keys())
        if not keys:
            return None
        key = keys[int(rng.integers(0, len(keys)))]
        cell = self.cells[key]
        if len(cell) == 1:
            return cell[0]
        g, w = self.config.depth_gamma, self.config.reuse_omega
        if g <= 0 and w <= 0:
            return cell[int(rng.integers(0, len(cell)))]
        weights = np.empty(len(cell), dtype=float)
        for i, eg in enumerate(cell):
            meta = eg.genome.metadata or {}
            depth = int(meta.get("depth", eg.genome.generation) or 0)
            reuse = int((reuse_counts or {}).get(eg.genome.genome_id, 0))
            weights[i] = max(((1.0 - g) ** depth) * ((1.0 - w) ** reuse), 1e-9)
        weights /= weights.sum()
        return cell[int(rng.choice(len(cell), p=weights))]

    # ── the book the harness scores against / persists ──
    def elites(self) -> list["EvaluatedGenome"]:
        """Union of every cell's elites (the diverse 'accepted book')."""
        return [eg for cell in self.cells.values() for eg in cell]

    def occupancy(self) -> dict[str, int]:
        return {"|".join(map(str, k)): len(v) for k, v in sorted(self.cells.items())}

    def qd_score(self) -> float:
        """Sum over cells of the best marginal-value in the cell (a QD-score proxy)."""
        total = 0.0
        for cell in self.cells.values():
            best = max((eg.fitness.objective.marginal_value or 0.0) for eg in cell)
            total += float(best)
        return total

    def n_cells(self) -> int:
        return len(self.cells)

    # ── persistence ──
    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "grid_dims": self.config.grid_dims,
                "cell_capacity": self.config.cell_capacity,
                "trend_edges": list(self.config.trend_edges),
                "speed_edges": list(self.config.speed_edges),
                "stress_edges": list(self.config.stress_edges),
                "depth_gamma": self.config.depth_gamma,
                "reuse_omega": self.config.reuse_omega,
            },
            "bin_edges": {k: list(v) for k, v in self.bin_edges.items()},
            "cells": {"|".join(map(str, k)): [eg.to_dict() for eg in v]
                      for k, v in self.cells.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QDArchive":
        from quant_fund_agent.agents.factor_research.evolution.controller import (
            EvaluatedGenome,
        )
        from quant_fund_agent.agents.factor_research.evolution.genome import Genome
        from quant_fund_agent.research_eval.fitness import FitnessResult

        cfgd = d.get("config", {})
        arch = cls(QDConfig(
            grid_dims=cfgd.get("grid_dims", 2),
            cell_capacity=cfgd.get("cell_capacity", 3),
            trend_edges=tuple(cfgd.get("trend_edges", (-0.33, 0.33))),
            speed_edges=tuple(cfgd.get("speed_edges", (0.5, 1.0))),
            stress_edges=tuple(cfgd.get("stress_edges", (0.9, 1.3))),
            depth_gamma=cfgd.get("depth_gamma", 0.0),
            reuse_omega=cfgd.get("reuse_omega", 0.0),
        ))
        if "bin_edges" in d:  # honour frozen edges exactly (never rebin)
            arch.bin_edges = {k: tuple(v) for k, v in d["bin_edges"].items()}
        for kstr, cell in d.get("cells", {}).items():
            key = tuple(int(x) for x in kstr.split("|"))
            arch.cells[key] = [
                EvaluatedGenome(genome=Genome.from_dict(e["genome"]),
                                fitness=FitnessResult.from_dict(e["fitness"]))
                for e in cell
            ]
        return arch
