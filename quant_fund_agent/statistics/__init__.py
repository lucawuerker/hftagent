"""Statistics package — base class, registry, schemas, auto-discovery."""

from quant_fund_agent.statistics._discover import discover_tests
from quant_fund_agent.statistics.base import BaseStatTest, StatTestContext
from quant_fund_agent.statistics.registry import (
    get_all_test_classes,
    get_mandatory_test_ids,
    get_optional_test_ids,
    get_test_class,
    instantiate_test,
    register_test,
)
from quant_fund_agent.statistics.schemas import StatTestResult, StatTestVerdict

__all__ = [
    "BaseStatTest",
    "StatTestContext",
    "StatTestResult",
    "StatTestVerdict",
    "register_test",
    "get_test_class",
    "get_all_test_classes",
    "get_mandatory_test_ids",
    "get_optional_test_ids",
    "instantiate_test",
    "discover_tests",
]
