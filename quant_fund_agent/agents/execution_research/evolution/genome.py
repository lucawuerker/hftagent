"""The execution-arm program type: one ``BaseExecutor`` source + its hypothesis.

Plugs into the factor arm's generic ``Genome`` container via the E1
``PROGRAM_TYPES`` registry (``Genome(program_type="executor", ...)``), so the
``EvolutionController``'s save/load, dedup fingerprints and lineage all work
unchanged.  ``factor_id`` is exposed as a property aliasing ``executor_id`` —
that is the one interface the container requires of a program.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from quant_fund_agent.agents.factor_research.evolution.genome import (
    register_program_type,
)


@dataclass
class ExecutionProgram:
    """One execution program: source code + the mechanism claim behind it.

    ``mechanism`` is the execution idea in words (e.g. "momentum decays in ~h
    bars — exit at 0.5·h under high vol"); ``expected_effect`` is the
    falsifiable claim checked against realised diagnostics (the execution
    analogue of ``FactorProgram.expected_sign``), e.g. "reduces turnover ≥20%
    at ≤10% gross-capture loss".
    """

    executor_id: str
    code: str
    name: str = ""
    regime: str = "per_underlying"          # which book shape it builds
    mechanism: str = ""
    expected_effect: str = ""
    description: str = ""

    @property
    def factor_id(self) -> str:
        """Genome-container interface: the program's lineage/dedup id."""
        return self.executor_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExecutionProgram":
        known = {f for f in cls.__dataclass_fields__}  # tolerate extra keys
        return cls(**{k: v for k, v in d.items() if k in known})


register_program_type("executor", ExecutionProgram)
