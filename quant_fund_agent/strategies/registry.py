"""Global strategy registry.

Mirrors the factor registry: every strategy class decorated with
``@register_strategy`` is stored here and can be looked up by
``strategy_id`` at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant_fund_agent.strategies.base import BaseStrategy

_STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {}


def register_strategy(cls: type[BaseStrategy]) -> type[BaseStrategy]:
    """Class decorator that registers a strategy implementation."""
    sid = getattr(cls, "strategy_id", None)
    if not sid:
        raise ValueError(
            f"{cls.__name__} must define a non-empty 'strategy_id' class attribute"
        )
    if sid in _STRATEGY_REGISTRY:
        raise ValueError(
            f"Duplicate strategy_id {sid!r}: {cls.__name__} vs "
            f"{_STRATEGY_REGISTRY[sid].__name__}"
        )
    _STRATEGY_REGISTRY[sid] = cls
    return cls


def get_strategy_class(strategy_id: str) -> type[BaseStrategy] | None:
    return _STRATEGY_REGISTRY.get(strategy_id)


def get_all_strategy_classes() -> dict[str, type[BaseStrategy]]:
    return dict(_STRATEGY_REGISTRY)


def instantiate_strategy(strategy_id: str) -> BaseStrategy:
    cls = _STRATEGY_REGISTRY.get(strategy_id)
    if cls is None:
        raise KeyError(f"Strategy {strategy_id!r} not found in registry")
    return cls()
