"""In-memory strategy database.

Stores ``StrategyRecord`` metadata that links to concrete
``BaseStrategy`` subclasses via the ``class_name`` field.
"""

from __future__ import annotations

import json
from pathlib import Path

from quant_fund_agent.schemas import StrategyRecord, StrategyStatus


class StrategyDatabase:
    """Registry of trading strategies."""

    def __init__(self) -> None:
        self._strategies: dict[str, StrategyRecord] = {}

    def add_strategy(self, strategy: StrategyRecord) -> None:
        self._strategies[strategy.id] = strategy

    def get_strategy(self, strategy_id: str) -> StrategyRecord | None:
        return self._strategies.get(strategy_id)

    def list_strategies(
        self, status: StrategyStatus | None = None
    ) -> list[StrategyRecord]:
        if status is None:
            return list(self._strategies.values())
        return [s for s in self._strategies.values() if s.status == status]

    def update_strategy(self, strategy: StrategyRecord) -> None:
        self._strategies[strategy.id] = strategy

    def remove_strategy(self, strategy_id: str) -> None:
        self._strategies.pop(strategy_id, None)

    # ----- persistence -----

    def save_to_json(self, path: str | Path) -> None:
        path = Path(path)
        payload = [s.model_dump(mode="json") for s in self._strategies.values()]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str))

    def load_from_json(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            return
        for raw in json.loads(path.read_text()):
            strategy = StrategyRecord.model_validate(raw)
            self._strategies[strategy.id] = strategy
