"""Registry for statistical tests — mirrors the factor / strategy pattern."""

from __future__ import annotations

from quant_fund_agent.statistics.base import BaseStatTest


_TEST_REGISTRY: dict[str, type[BaseStatTest]] = {}


def register_test(cls: type[BaseStatTest]) -> type[BaseStatTest]:
    """Decorator that registers a ``BaseStatTest`` subclass by its ``test_id``."""
    if not cls.test_id:
        raise ValueError(f"{cls.__name__}.test_id must be set")
    if cls.test_id in _TEST_REGISTRY:
        raise ValueError(
            f"Duplicate test_id '{cls.test_id}' "
            f"({cls.__name__} vs {_TEST_REGISTRY[cls.test_id].__name__})"
        )
    _TEST_REGISTRY[cls.test_id] = cls
    return cls


def get_test_class(test_id: str) -> type[BaseStatTest]:
    return _TEST_REGISTRY[test_id]


def instantiate_test(test_id: str) -> BaseStatTest:
    return get_test_class(test_id)()


def get_all_test_classes() -> dict[str, type[BaseStatTest]]:
    return dict(_TEST_REGISTRY)


def get_mandatory_test_ids() -> list[str]:
    return [tid for tid, cls in _TEST_REGISTRY.items() if cls.is_mandatory]


def get_optional_test_ids() -> list[str]:
    return [tid for tid, cls in _TEST_REGISTRY.items() if not cls.is_mandatory]
