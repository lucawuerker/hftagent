"""Factor package – base class, registry, and auto-discovery."""

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import (
    get_all_factor_classes,
    get_factor_class,
    instantiate_factor,
    register_factor,
)
from quant_fund_agent.factors._discover import discover_factors

__all__ = [
    "BaseFactor",
    "register_factor",
    "get_factor_class",
    "get_all_factor_classes",
    "instantiate_factor",
    "discover_factors",
]
