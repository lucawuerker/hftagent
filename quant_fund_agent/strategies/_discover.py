"""Auto-discovery of strategy modules."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path


def discover_strategies() -> None:
    package_dir = Path(__file__).resolve().parent
    package_name = "quant_fund_agent.strategies"

    for _finder, module_name, _ispkg in pkgutil.walk_packages(
        [str(package_dir)], prefix=f"{package_name}."
    ):
        if module_name.endswith("_discover"):
            continue
        importlib.import_module(module_name)
