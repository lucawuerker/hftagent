"""Deterministic research-time evaluation for the evolutionary factor researcher.

This package is the **fitness function** half of the evolutionary loop designed in
``docs/research-evolution/DESIGN.md``.  The single most important principle is that
*the LLM ideates and mutates; a deterministic harness scores* — so everything here
is deterministic, un-gameable and out-of-sample by construction, and nothing in it
ever consults an LLM.

Modules:

* :mod:`~quant_fund_agent.research_eval.splits` — the IS/VAL/TEST temporal split,
  Combinatorial Purged Cross-Validation (CPCV, with purge + embargo) and the
  walk-forward wrapper.  This is the overfitting-control seam built first.
* :mod:`~quant_fund_agent.research_eval.deflation` — deflated-IC haircut and the
  Bailey–López de Prado Deflated Sharpe Ratio, both *N_trials*-aware so the search
  itself is treated as the multiple-testing machine it is.
* :mod:`~quant_fund_agent.research_eval.fitness` — the CORE Pareto objective vector,
  the hard gates and the Pareto-dominance primitives.
* :mod:`~quant_fund_agent.research_eval.harness` — orchestration that turns a
  candidate signal (+ the current book) into a :class:`fitness.FitnessResult`,
  reusing the trusted ``comparison/`` and ``backtesting/`` code.
"""

from __future__ import annotations

from quant_fund_agent.research_eval.deflation import (
    deflated_ic,
    deflated_sharpe_ratio,
    expected_max_sharpe,
)
from quant_fund_agent.research_eval.fitness import (
    GateResults,
    ObjectiveVector,
    FitnessResult,
    dominates,
    non_dominated_front,
)
from quant_fund_agent.research_eval.splits import (
    CPCVFold,
    ThreeWaySplit,
    WalkForwardFold,
    cpcv_folds,
    three_way_split,
    walk_forward_folds,
)

__all__ = [
    "ThreeWaySplit",
    "CPCVFold",
    "WalkForwardFold",
    "three_way_split",
    "cpcv_folds",
    "walk_forward_folds",
    "deflated_ic",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "ObjectiveVector",
    "GateResults",
    "FitnessResult",
    "dominates",
    "non_dominated_front",
]
