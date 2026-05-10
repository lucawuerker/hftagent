"""Paper database with JSON persistence and lookahead-bias filtering.

Papers are stored as a single JSON index file.  Each entry carries a
``published_date`` so that downstream consumers (factor research, backtest
engine) can restrict themselves to papers published before a given cutoff
date, preventing lookahead bias.

Recommended storage layout::

    data/papers/
    ├── index.json          ← metadata for every paper (this DB loads/saves it)
    └── pdfs/
        ├── black_scholes_1973.pdf
        └── …
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from quant_fund_agent.schemas import Paper, PaperStatus


class PaperDatabase:
    """Registry of academic / research papers."""

    def __init__(self) -> None:
        self._papers: dict[str, Paper] = {}

    # ----- CRUD -----

    def add_paper(self, paper: Paper) -> None:
        self._papers[paper.id] = paper

    def get_paper(self, paper_id: str) -> Paper | None:
        return self._papers.get(paper_id)

    def list_papers(self, status: PaperStatus | None = None) -> list[Paper]:
        if status is None:
            return list(self._papers.values())
        return [p for p in self._papers.values() if p.status == status]

    def mark_paper_status(self, paper_id: str, status: PaperStatus) -> None:
        if paper_id in self._papers:
            self._papers[paper_id].status = status

    def remove_paper(self, paper_id: str) -> None:
        self._papers.pop(paper_id, None)

    # ----- lookahead-bias guard -----

    def list_papers_before(
        self,
        cutoff: date,
        status: PaperStatus | None = None,
    ) -> list[Paper]:
        """Return only papers published strictly before *cutoff*.

        Papers without a ``published_date`` are excluded to be safe.
        """
        papers = self.list_papers(status=status)
        return [
            p for p in papers
            if p.published_date is not None and p.published_date < cutoff
        ]

    # ----- persistence -----

    def save_to_json(self, path: str | Path) -> None:
        path = Path(path)
        payload = [p.model_dump(mode="json") for p in self._papers.values()]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str))

    def load_from_json(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            return
        for raw in json.loads(path.read_text()):
            paper = Paper.model_validate(raw)
            self._papers[paper.id] = paper
