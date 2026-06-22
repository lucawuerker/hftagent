"""Ingest raw LOBSTER message/orderbook CSVs into the ``ticker_data`` format.

Two concerns live here:

* :mod:`converter` — turn a day's raw ``*_message_*`` + ``*_orderbook_*`` pair
  into the 19-column, 10-second-bar CSV that
  :mod:`quant_fund_agent.backtesting.data_loader` consumes.  Levels-agnostic
  (level 3, 5, 10 … inferred from the orderbook column count), streams one day
  at a time so RAM stays bounded by a single day's raw files.
* :mod:`orchestrator` + :mod:`lobster_portal` — drive the LOBSTER web portal
  with Playwright, download a chunk, convert it, then delete the raw files so a
  multi-year, multi-ticker pull fits inside a tiny (<50 GB) disk budget.

The reconstructed column definitions are documented in
``docs/lobster-ingestion/CONVERSION_SPEC.md`` (the original 2019 aggregator is
not in the repo, so these are reverse-engineered from the LOBSTER spec).
"""

from quant_fund_agent.data.lobster_ingest.converter import (
    OUTPUT_COLUMNS,
    convert_archive,
    convert_day,
    convert_folder,
    convert_sampled_day,
    discover_day_files,
)

__all__ = [
    "OUTPUT_COLUMNS",
    "convert_archive",
    "convert_day",
    "convert_folder",
    "convert_sampled_day",
    "discover_day_files",
]
