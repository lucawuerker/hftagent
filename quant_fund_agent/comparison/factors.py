"""Load a prerun's factor ids and filter them to what's computable *now*.

The whole comparison must run on the data that's available today (the 51-ticker
LOBSTER sample) and re-run unchanged once the full LOBSTER universe / FMP
fundamentals are downloaded.  A researched factor that declares a field the
current panel doesn't have (e.g. ``peRatio``) simply can't be computed yet —
:func:`usable_factor_ids` drops it (with a reason) instead of crashing, and the
report shows the usable count per prerun.  When the data lands, the same factor
lights up with no code change.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from quant_fund_agent.databases import FactorDatabase
from quant_fund_agent.factors import preruns
from quant_fund_agent.schemas import FactorRecord, FactorSource

log = logging.getLogger("comparison.factors")


def prerun_factor_records(name: str) -> list[FactorRecord]:
    """The RESEARCHER FactorRecords held in a prerun's own factor DB."""
    db = FactorDatabase()
    db.load_from_json(preruns.db_path(name))
    return db.list_factors(source=FactorSource.RESEARCHER)


def prerun_factor_ids(
    name: str, include_seeds: bool = False, base_db: Path | None = None,
) -> list[str]:
    """Factor ids contributed by a prerun: its researcher factors (± seeds).

    ``include_seeds`` prepends the SEED alphas from ``base_db`` (the global
    library) so a comparison can optionally evaluate seeds + researched factors.
    Order-preserving and de-duplicated.
    """
    ids: list[str] = []
    if include_seeds:
        base = FactorDatabase()
        base.load_from_json(base_db or preruns.BASE_FACTOR_DB)
        ids += [f.id for f in base.list_factors(source=FactorSource.SEED)]
    ids += [f.id for f in prerun_factor_records(name)]

    seen: set[str] = set()
    out: list[str] = []
    for fid in ids:
        if fid not in seen:
            seen.add(fid)
            out.append(fid)
    return out


def factor_names(names: list[str]) -> dict[str, str]:
    """Best-effort ``{factor_id: display name}`` across the given preruns."""
    out: dict[str, str] = {}
    for name in names:
        for rec in prerun_factor_records(name):
            out.setdefault(rec.id, rec.name or rec.id)
    return out


def load_panel_cached(data_dir: str | None = None, n_tickers: int | None = None) -> dict[str, Any]:
    """Load (and module-cache) the factor panel the whole comparison shares.

    Sets ``DATA_DIR`` / ``ARCHITECT_N_TICKERS`` so the modeling service — whose
    ``_load_panel`` / ``_factor_signal`` every track reuses — reads the intended
    data and universe.  ``DATA_DIR`` is captured at import in the service, so we
    also override its module global defensively in case it imported earlier.
    """
    from quant_fund_agent.factors import discover_factors
    from quant_fund_agent.modeling import service

    if data_dir:
        os.environ["DATA_DIR"] = data_dir
        service.DATA_DIR = data_dir
    if n_tickers is not None:
        os.environ["ARCHITECT_N_TICKERS"] = str(n_tickers)

    discover_factors()
    return service._load_panel()


def usable_factor_ids(
    factor_ids: list[str], panel: dict[str, Any],
) -> tuple[list[str], dict[str, str]]:
    """Split ids into those computable on ``panel`` now and those that aren't.

    Probes each factor's signal via the shared, cached
    ``modeling.service._factor_signal`` (so later tracks reuse the result).  A
    factor is dropped if its class isn't registered (code missing), its ``calc``
    raises (typically a field the current panel lacks), or it yields an all-NaN
    signal.  Returns ``(usable_ids, {dropped_id: reason})``.
    """
    from quant_fund_agent.factors import get_factor_class
    from quant_fund_agent.modeling.service import _factor_signal

    usable: list[str] = []
    dropped: dict[str, str] = {}
    for fid in factor_ids:
        if get_factor_class(fid) is None:
            dropped[fid] = "no factor class registered (generated code missing)"
            continue
        try:
            sig = _factor_signal(fid, panel)
        except Exception as e:  # noqa: BLE001 — a bad factor must not abort the sweep
            dropped[fid] = f"calc failed: {type(e).__name__}: {e}"[:200]
            continue
        try:
            all_nan = bool(sig.isna().all().all())
        except Exception:
            all_nan = sig is None
        if all_nan:
            dropped[fid] = "all-NaN signal on the current data"
            continue
        usable.append(fid)

    if dropped:
        log.info("usable_factor_ids: %d usable, %d dropped (%s)",
                 len(usable), len(dropped),
                 ", ".join(list(dropped)[:5]) + ("…" if len(dropped) > 5 else ""))
    return usable, dropped
