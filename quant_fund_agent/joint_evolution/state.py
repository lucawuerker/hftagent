"""The shared SOTA state the two arms evolve against (J0).

RD-Agent(Q)'s central object, adapted: instead of one SOTA factor + one SOTA
model, we hold the **current curated factor book** (the factor arm's accepted
Pareto book) and the **current SOTA executor** (the exec arm's best archive
member), plus the version pointer of the frozen evaluation signals that tie
them together.  Persisted as JSON in ``<scope>/joint/joint_state.json``
(alongside the ledger + scheduler posterior) so a joint run is resumable at
any block boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def book_hash(book: list[dict[str, Any]]) -> str:
    """Whitespace-insensitive fingerprint of a factor book (id + code)."""
    payload = "\n".join(
        f"{p.get('factor_id')}:{''.join(str(p.get('code', '')).split())}"
        for p in book
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class SOTAState:
    """Everything the outer layer knows between blocks."""

    book: list[dict[str, Any]] = field(default_factory=list)
    #: {"factor_id", "code"} programs — the factor arm's current accepted book
    book_hash: str = ""
    sota_executor: dict[str, Any] | None = None
    #: {"executor_id", "code", "regime", "genome_id", "objective"} or None
    frozen_signals_version: int = 0          # 0 = nothing frozen yet
    frozen_signals_manifest: str = ""
    block_index: int = 0                     # blocks completed so far
    metadata: dict[str, Any] = field(default_factory=dict)

    def set_book(self, book: list[dict[str, Any]]) -> None:
        self.book = list(book)
        self.book_hash = book_hash(self.book)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SOTAState":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "SOTAState":
        return cls.from_dict(json.loads(Path(path).read_text()))
