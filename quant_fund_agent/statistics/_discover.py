"""Auto-discover statistical-test modules so their @register_test runs."""

from __future__ import annotations

import importlib
import pkgutil


def discover_tests() -> None:
    """Import every module under ``quant_fund_agent.statistics.tests``.

    Idempotent — safe to call multiple times.
    """
    pkg = importlib.import_module("quant_fund_agent.statistics.tests")
    for mod in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"quant_fund_agent.statistics.tests.{mod.name}")
