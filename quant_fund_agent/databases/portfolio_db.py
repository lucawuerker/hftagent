"""Append-only registry of Portfolio Manager decisions.

Every invocation of the PM agent produces one ``PortfolioRecord``.
Storing them is what lets us:

* compare allocations across days / weeks / regimes,
* run multiple PMs in parallel and audit each one independently
  (filter by ``pm_name``),
* roll forward a simulation by always reading the *latest* allocation.

The DB is intentionally minimal — it's a journal, not a query engine.
"""

from __future__ import annotations

import json
from pathlib import Path

from quant_fund_agent.schemas import PortfolioRecord


class PortfolioDatabase:
    """Append-only registry of PortfolioRecord snapshots."""

    def __init__(self) -> None:
        self._portfolios: dict[str, PortfolioRecord] = {}

    def add_portfolio(self, portfolio: PortfolioRecord) -> None:
        self._portfolios[portfolio.id] = portfolio

    def get_portfolio(self, portfolio_id: str) -> PortfolioRecord | None:
        return self._portfolios.get(portfolio_id)

    def list_portfolios(self, pm_name: str | None = None) -> list[PortfolioRecord]:
        out = list(self._portfolios.values())
        if pm_name is not None:
            out = [p for p in out if p.pm_name == pm_name]
        out.sort(key=lambda p: p.created_at)
        return out

    def latest(self, pm_name: str | None = None) -> PortfolioRecord | None:
        portfolios = self.list_portfolios(pm_name=pm_name)
        return portfolios[-1] if portfolios else None

    # ── persistence ───────────────────────────────────────────────────

    def save_to_json(self, path: str | Path) -> None:
        path = Path(path)
        payload = [p.model_dump(mode="json") for p in self._portfolios.values()]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str))

    def load_from_json(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            return
        for raw in json.loads(path.read_text()):
            portfolio = PortfolioRecord.model_validate(raw)
            self._portfolios[portfolio.id] = portfolio
