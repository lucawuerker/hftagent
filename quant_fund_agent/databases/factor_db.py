"""In-memory factor / feature database.

Stores ``FactorRecord`` metadata, trading ideas, and maintains a
correlation matrix across all factors.  The ``class_name`` field on each
record links to the concrete ``BaseFactor`` subclass in the factor
registry, so the backtesting engine can instantiate the right
implementation on demand.

Factors carry a ``source`` tag (``SEED`` vs ``RESEARCHER``) so the
simulation can start from a clean, deterministic baseline (seed factors
only) and so factors invented by the Factor Researcher Agent can be
filtered, audited, or purged in bulk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_fund_agent.schemas import FactorRecord, FactorSource, TradingIdea


class FactorDatabase:
    """Registry of factors, trading ideas, and their cross-correlations."""

    def __init__(self) -> None:
        self._factors: dict[str, FactorRecord] = {}
        self._trading_ideas: dict[str, TradingIdea] = {}
        self._correlation_matrix: pd.DataFrame = pd.DataFrame()

    # ----- factors -----

    def add_factor(self, factor: FactorRecord) -> None:
        self._factors[factor.id] = factor

    def get_factor(self, factor_id: str) -> FactorRecord | None:
        return self._factors.get(factor_id)

    def list_factors(
        self,
        source: FactorSource | None = None,
        research_session_id: str | None = None,
    ) -> list[FactorRecord]:
        """Return factors, optionally filtered by source and/or session.

        - ``source=FactorSource.SEED`` returns just the hard-coded baseline.
        - ``source=FactorSource.RESEARCHER`` returns just agent-generated
          factors (optionally narrowed to a single ``research_session_id``).
        - ``source=None`` returns everything.
        """
        factors = list(self._factors.values())
        if source is not None:
            factors = [f for f in factors if f.source == source]
        if research_session_id is not None:
            factors = [f for f in factors if f.research_session_id == research_session_id]
        return factors

    def remove_factor(self, factor_id: str) -> None:
        self._factors.pop(factor_id, None)

    def purge_researcher_factors(
        self,
        research_session_id: str | None = None,
    ) -> list[str]:
        """Remove all researcher-generated factors (or a single session).

        Use this between simulation runs so the next 1-year backtest
        starts from the seed ensemble only, with no memory of factors
        invented in previous runs.

        Returns the list of removed factor IDs.
        """
        to_remove = [
            fid for fid, f in self._factors.items()
            if f.source == FactorSource.RESEARCHER
            and (research_session_id is None or f.research_session_id == research_session_id)
        ]
        for fid in to_remove:
            self._factors.pop(fid, None)
        return to_remove

    # ----- trading ideas -----

    def add_trading_idea(self, idea: TradingIdea) -> None:
        self._trading_ideas[idea.id] = idea

    def get_trading_idea(self, idea_id: str) -> TradingIdea | None:
        return self._trading_ideas.get(idea_id)

    def list_trading_ideas(self) -> list[TradingIdea]:
        return list(self._trading_ideas.values())

    # ----- correlation matrix -----

    def update_correlation_matrix(self, matrix: pd.DataFrame) -> None:
        self._correlation_matrix = matrix

    def get_correlation_matrix(self) -> pd.DataFrame:
        return self._correlation_matrix

    def get_factor_correlations(self, factor_id: str) -> pd.Series | None:
        if factor_id in self._correlation_matrix.index:
            return self._correlation_matrix.loc[factor_id]
        return None

    # ----- persistence -----

    def save_to_json(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "factors": [f.model_dump(mode="json") for f in self._factors.values()],
            "trading_ideas": [t.model_dump(mode="json") for t in self._trading_ideas.values()],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str))

    def load_from_json(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            return
        payload = json.loads(path.read_text())
        for raw in payload.get("factors", []):
            f = FactorRecord.model_validate(raw)
            self._factors[f.id] = f
        for raw in payload.get("trading_ideas", []):
            t = TradingIdea.model_validate(raw)
            self._trading_ideas[t.id] = t
