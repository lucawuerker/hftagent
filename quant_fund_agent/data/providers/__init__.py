"""Data-provider registry.

Providers register their class here keyed by ``name`` so config can select one by
string (``data.provider``).  Add new vendors to ``PROVIDERS`` — see
``docs/data-layer/DATA_PROVIDERS.md``.
"""

from __future__ import annotations

from quant_fund_agent.data.providers.base import DataProvider
from quant_fund_agent.data.providers.lobster import LobsterProvider
from quant_fund_agent.data.providers.yfinance import YFinanceProvider

# name → provider class.  fmp / alphavantage land here in Phase 4.
PROVIDERS: dict[str, type[DataProvider]] = {
    LobsterProvider.name: LobsterProvider,
    YFinanceProvider.name: YFinanceProvider,
}


def get_provider_class(name: str) -> type[DataProvider]:
    cls = PROVIDERS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown data provider {name!r}. "
            f"Available: {sorted(PROVIDERS)}."
        )
    return cls


__all__ = [
    "DataProvider", "LobsterProvider", "YFinanceProvider",
    "PROVIDERS", "get_provider_class",
]
