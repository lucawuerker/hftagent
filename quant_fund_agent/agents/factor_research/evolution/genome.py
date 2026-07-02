"""The unit of evolution: a factor program, or a set of them.

Per the locked design decision 4 (``--evolution-unit {single,set}``):

* ``SINGLE`` — the genome is **one** factor program (source code + the hypothesis
  metadata the LLM declared for it).  Fitness measures what the program adds to
  the current Pareto archive (LOCO marginal value).
* ``SET`` — the genome is a whole "alpha program": a list of factor programs
  evaluated **jointly** (the set's own combined-model OOS IC is the primary
  axis).  Closest to AlphaEvolve.  The dataclass supports it now; the SET
  operators/wiring land in P5.

Everything is a plain serialisable dataclass — genomes are logged to the lineage
file and must survive a JSON round-trip so runs are resumable and auditable.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FactorProgram:
    """One factor program: source code + the hypothesis metadata behind it.

    ``expected_sign`` (+1/-1, or ``None`` when the LLM declared none) is the
    falsifiable direction claim that feeds the harness's sign-consistency check —
    the "free overfit detector you only get because an LLM stated the hypothesis".
    """

    factor_id: str
    code: str
    name: str = ""
    category: str = "other"
    trading_idea: str = ""
    description: str = ""
    prediction_horizon: int = 6
    suggested_horizons: list[int] = field(default_factory=list)
    expected_sign: int | None = None
    source_paper_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FactorProgram":
        known = {f for f in cls.__dataclass_fields__}  # tolerate extra keys
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Genome:
    """A member of the evolving population.

    ``programs`` has exactly one element in SINGLE mode.  ``parent_ids`` /
    ``operator`` / ``generation`` / ``island`` are the lineage fields the thesis
    analysis needs (who mutated from whom, under which operator, when, where).
    """

    genome_id: str
    programs: list[FactorProgram]
    unit: str = "single"                    # "single" | "set"
    generation: int = 0
    island: int = 0
    parent_ids: list[str] = field(default_factory=list)
    operator: str = "seed"                  # "seed" | "llm_semantic" | "jitter" | "crossover" | ...
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.unit not in ("single", "set"):
            raise ValueError(f"unit must be 'single' or 'set', got {self.unit!r}")
        if not self.programs:
            raise ValueError("a genome needs at least one factor program")
        if self.unit == "single" and len(self.programs) != 1:
            raise ValueError(
                f"a SINGLE genome holds exactly one program (got {len(self.programs)})")

    @property
    def program(self) -> FactorProgram:
        """The sole program of a SINGLE genome (the common access path)."""
        if self.unit != "single":
            raise ValueError("`.program` is only defined for SINGLE genomes")
        return self.programs[0]

    @property
    def factor_ids(self) -> list[str]:
        return [p.factor_id for p in self.programs]

    def code_fingerprint(self) -> str:
        """Stable hash of the member programs' source — the dedup key.

        Whitespace-insensitive per line, and the program's own ``factor_id`` is
        masked out — so an LLM re-emitting the same program under a fresh id (or
        with different indentation/blank lines) is still recognised as identical.
        """
        h = hashlib.sha256()
        for p in sorted(self.programs, key=lambda p: p.factor_id):
            body = p.code.replace(p.factor_id, "<FID>") if p.factor_id else p.code
            normalised = "\n".join(
                line.strip() for line in body.splitlines() if line.strip())
            h.update(normalised.encode())
        return h.hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["programs"] = [p.to_dict() for p in self.programs]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Genome":
        d = dict(d)
        d["programs"] = [FactorProgram.from_dict(p) for p in d.get("programs", [])]
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
