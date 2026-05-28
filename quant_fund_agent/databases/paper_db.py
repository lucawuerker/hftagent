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
import re
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

    # ----- auto-discovery -----

    @staticmethod
    def _slugify(name: str) -> str:
        """Filename → slug used as a default paper id."""
        stem = Path(name).stem.lower()
        stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
        return stem or "paper"

    def auto_discover_pdfs(
        self,
        pdf_dir: str | Path,
        index_path: str | Path | None = None,
        default_published_date: date | None = None,
    ) -> list[Paper]:
        """Scan ``pdf_dir`` for PDFs and register any that aren't tracked.

        Filenames are used as a default ``title`` and slugged to produce
        the paper ``id``.  Metadata is intentionally minimal — the
        Factor Researcher Agent fills in the abstract / key ideas the
        first time it actually reads the paper.

        If ``index_path`` is provided the updated index is persisted.

        Returns the list of newly-added ``Paper`` objects.
        """
        pdf_dir = Path(pdf_dir)
        if not pdf_dir.exists():
            return []

        # Build a lookup of already-tracked file_paths so we don't double-add.
        tracked = {
            (Path(p.file_path).name if p.file_path else "")
            for p in self._papers.values()
        }

        new_papers: list[Paper] = []
        for pdf in sorted(pdf_dir.glob("*.pdf")):
            if pdf.name in tracked:
                continue
            paper_id = self._slugify(pdf.name)
            # Avoid collision with an existing id (e.g. two files slug
            # to the same thing).
            suffix = 1
            candidate = paper_id
            while candidate in self._papers:
                suffix += 1
                candidate = f"{paper_id}_{suffix}"
            paper = Paper(
                id=candidate,
                title=pdf.stem,
                file_path=f"pdfs/{pdf.name}",
                published_date=default_published_date,
            )
            self._papers[paper.id] = paper
            new_papers.append(paper)

        if index_path is not None and new_papers:
            self.save_to_json(index_path)
        return new_papers
