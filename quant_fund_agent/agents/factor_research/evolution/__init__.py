"""Closed-loop evolutionary factor research (FunSearch/AlphaEvolve spirit).

The LLM is the *mutation operator*; the deterministic
:mod:`quant_fund_agent.research_eval` harness is the *fitness function*; this
package is the controller in between.  See ``docs/research-evolution/DESIGN.md``.

Modules:

* :mod:`~quant_fund_agent.agents.factor_research.evolution.genome` — the unit of
  evolution: one factor program (SINGLE) or a set of them ("alpha program", SET).
* :mod:`~quant_fund_agent.agents.factor_research.evolution.controller` — the
  deterministic evolution controller: population + islands, constrained NSGA-II
  (non-dominated sort + crowding distance, gate-aware), the Pareto archive that
  doubles as the "accepted book", the ``N_trials`` counter feeding deflation, and
  the full lineage log.
* :mod:`~quant_fund_agent.agents.factor_research.evolution.mutation` — the
  mutation operators: LLM-semantic (the creative one), programmatic window-jitter
  (cheap; doubles as the plateau-penalty probe) and LLM crossover.
* :mod:`~quant_fund_agent.agents.factor_research.evolution.reflection` — turns a
  candidate's diagnostics dict into the natural-language mutation brief the next
  generation's LLM sees (the teacher channel).
* :mod:`~quant_fund_agent.agents.factor_research.evolution.loop` — runs the
  generations: seed → select parents → mutate → compile → evaluate → insert.
"""

from __future__ import annotations

from quant_fund_agent.agents.factor_research.evolution.genome import FactorProgram, Genome
from quant_fund_agent.agents.factor_research.evolution.controller import (
    EvaluatedGenome,
    EvolutionController,
    crowding_distance,
    non_dominated_sort,
)

__all__ = [
    "FactorProgram",
    "Genome",
    "EvaluatedGenome",
    "EvolutionController",
    "non_dominated_sort",
    "crowding_distance",
]
