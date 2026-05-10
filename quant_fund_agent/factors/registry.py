"""Global factor registry.

Every factor class decorated with ``@register_factor`` is stored here
and can be looked up by ``factor_id`` at runtime.  The backtesting engine
and the databases use this registry to instantiate factors on demand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant_fund_agent.factors.base import BaseFactor

_FACTOR_REGISTRY: dict[str, type[BaseFactor]] = {}


def register_factor(cls: type[BaseFactor]) -> type[BaseFactor]:
    """Class decorator that registers a factor implementation."""
    fid = getattr(cls, "factor_id", None)
    if not fid:
        raise ValueError(
            f"{cls.__name__} must define a non-empty 'factor_id' class attribute"
        )
    if fid in _FACTOR_REGISTRY:
        raise ValueError(
            f"Duplicate factor_id {fid!r}: {cls.__name__} vs "
            f"{_FACTOR_REGISTRY[fid].__name__}"
        )
    _FACTOR_REGISTRY[fid] = cls
    return cls


def get_factor_class(factor_id: str) -> type[BaseFactor] | None:
    return _FACTOR_REGISTRY.get(factor_id)


def get_all_factor_classes() -> dict[str, type[BaseFactor]]:
    return dict(_FACTOR_REGISTRY)


def instantiate_factor(factor_id: str) -> BaseFactor:
    cls = _FACTOR_REGISTRY.get(factor_id)
    if cls is None:
        raise KeyError(f"Factor {factor_id!r} not found in registry")
    return cls()
