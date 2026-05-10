"""Strategy package – base class, registry, and auto-discovery."""

from quant_fund_agent.strategies.base import BaseStrategy
from quant_fund_agent.strategies.registry import (
    get_all_strategy_classes,
    get_strategy_class,
    instantiate_strategy,
    register_strategy,
)
from quant_fund_agent.strategies._discover import discover_strategies

__all__ = [
    "BaseStrategy",
    "register_strategy",
    "get_strategy_class",
    "get_all_strategy_classes",
    "instantiate_strategy",
    "discover_strategies",
]
