"""One-time bulk ingestion of the FMP **premium** archive.

This package downloads a local, immutable, resumable copy of everything FMP
serves for a point-in-time index universe — full OHLCV (adjusted *and*
unadjusted) plus the complete fundamental record — and stores it verbatim as
parquet under ``data/vendor/fmp/``.

It is the *acquisition* half of the premium path documented in
``docs/data-layer/SP500_MEMBERSHIP.md`` §8.  The *consumption* half is
:mod:`quant_fund_agent.data.providers.fmp_archive`, which reads this archive
offline and emits the standard panel contract.

Layering (deliberate — richness must not be bounded by today's vocabulary):

* **Layer A — raw archive.**  Every vendor field is kept as returned, one
  parquet per ``(endpoint, symbol)``.  Adding a canonical field later needs no
  refetch.
* **Layer B — panel wiring.**  The archive provider maps a *curated* subset of
  those raw fields onto canonical panel field names with point-in-time
  availability stamping.

Modules mirror ``data/lobster_ingest/``: :mod:`endpoints` (what to fetch),
:mod:`client` (rate-limited HTTP), :mod:`store` (where it lands + resume state),
:mod:`capabilities` (what the plan actually serves), :mod:`symbols` (delisted
name resolution) and :mod:`download` (orchestration).
"""

from __future__ import annotations

from quant_fund_agent.data.fmp_ingest.client import (
    FMPClient,
    FetchResult,
    RateLimiter,
)
from quant_fund_agent.data.fmp_ingest.endpoints import (
    DEFAULT_GROUPS,
    ENDPOINTS,
    Endpoint,
    endpoints_for_groups,
    known_groups,
)
from quant_fund_agent.data.fmp_ingest.store import Archive, ManifestEntry

__all__ = [
    "Archive",
    "DEFAULT_GROUPS",
    "ENDPOINTS",
    "Endpoint",
    "FMPClient",
    "FetchResult",
    "ManifestEntry",
    "RateLimiter",
    "endpoints_for_groups",
    "known_groups",
]
