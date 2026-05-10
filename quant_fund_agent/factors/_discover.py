"""Auto-discovery of factor modules.

Calling ``discover_factors()`` walks every sub-package under
``quant_fund_agent.factors`` and imports it, which triggers the
``@register_factor`` decorators and populates the global registry.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path


def discover_factors() -> None:
    package_dir = Path(__file__).resolve().parent
    package_name = "quant_fund_agent.factors"

    for _finder, module_name, _ispkg in pkgutil.walk_packages(
        [str(package_dir)], prefix=f"{package_name}."
    ):
        if module_name.endswith("_discover"):
            continue
        importlib.import_module(module_name)
