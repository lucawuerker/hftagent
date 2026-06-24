"""Load a prerun's factor ids and filter them to what's computable *now*.

The whole comparison must run on the active provider panel (LOBSTER, yfinance,
FMP, AlphaVantage, ...).  A researched factor that declares a field the current
panel doesn't have (e.g. ``peRatio`` on a plain OHLCV provider) simply can't be
computed yet —
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

SEED_BASELINE_NAMES = {"main", "seed", "seeds"}


def is_seed_baseline(name: str) -> bool:
    """True for comparison aliases that mean the built-in seed factor library."""
    return name.strip().lower() in SEED_BASELINE_NAMES


def seed_factor_records(base_db: Path | None = None) -> list[FactorRecord]:
    """The hard-coded seed factors from the main factor DB."""
    db = FactorDatabase()
    db.load_from_json(base_db or preruns.BASE_FACTOR_DB)
    return db.list_factors(source=FactorSource.SEED)


def prerun_factor_records(name: str) -> list[FactorRecord]:
    """FactorRecords for a comparison input: seed baseline or a prerun DB."""
    if is_seed_baseline(name):
        return seed_factor_records()
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
        ids += [f.id for f in seed_factor_records(base_db)]
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


def prediction_horizons(names: list[str]) -> dict[str, int]:
    """Best-effort ``{factor_id: prediction_horizon}`` across the given preruns.

    Drives the per-factor IC headline and the brute-force combined-signal horizon
    (mode of these).  Records default to 6, so a prerun never seen by the horizon
    migration still resolves to a sensible value.
    """
    out: dict[str, int] = {}
    for name in names:
        for rec in prerun_factor_records(name):
            out.setdefault(rec.id, int(getattr(rec, "prediction_horizon", 6) or 6))
    return out


def _subsample_bars(panel: dict[str, Any], max_bars: int | None) -> dict[str, Any]:
    """Uniformly stride every field frame to at most ``max_bars`` timestamps.

    The intraday LOBSTER panel is ~1.4M bars/ticker; striding the *shared* index
    to a few thousand bars is the single biggest speed lever — it slims every
    track at once (IC, analytics, brute-force) and removes the brute-force OOM,
    since feature matrices are built from this slim panel.  Every field is sliced
    to the *same* retained index so the panel stays aligned.  Horizons are then in
    retained-bar units (consistent across preruns).  ``None`` / already-small
    panels are a no-op.
    """
    if not max_bars or not panel:
        return panel
    index = next(iter(panel.values())).index
    n = len(index)
    if n <= max_bars:
        return panel
    import numpy as np

    keep = index[np.linspace(0, n - 1, max_bars, dtype=int)]
    keep = keep[~keep.duplicated()]
    log.info("subsampling panel: %d → %d bars (max_bars=%d)", n, len(keep), max_bars)
    return {k: df.loc[keep] for k, df in panel.items()}


def _restrict_to_periods(
    panel: dict[str, Any], is_window: str | None, oos_window: str | None,
) -> dict[str, Any]:
    """Restrict every field frame to the union of the train + OOS calendar windows.

    When the comparison uses a date-based IS/OOS split, the whole comparison should
    run on just those months (so the IC / analytics / brute-force tracks are all
    scored on the same windowed data).  ``None`` for either window (the ratio-split
    default) is a no-op.
    """
    if not panel or not (is_window and oos_window):
        return panel
    from quant_fund_agent.comparison.config import period_mask

    index = next(iter(panel.values())).index
    mask = period_mask(is_window, index) | period_mask(oos_window, index)
    kept = int(mask.sum())
    log.info("restricting panel to train∪OOS windows: %d → %d bars", len(index), kept)
    if kept == 0:
        log.warning("date windows select 0 bars — check the months lie within the data "
                    "range (%s … %s); leaving the panel unrestricted.", index.min(), index.max())
        return panel
    return {k: df[mask] for k, df in panel.items()}


def load_panel_cached(
    data_dir: str | None = None,
    n_tickers: int | None = None,
    max_bars: int | None = None,
    *,
    tickers: list[str] | None = None,
    is_window: str | None = None,
    oos_window: str | None = None,
    data_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load (and cache) the factor panel the whole comparison shares.

    The loaded panel is injected into ``modeling.service._PANEL_CACHE`` so the
    brute-force track's ``evaluate_config`` / ``fit_and_backtest`` and the IC track's
    ``backtest_factor`` all use the same panel and universe.
    """
    from quant_fund_agent.config import get_settings
    from quant_fund_agent.data import load_panel
    from quant_fund_agent.factors import discover_factors
    from quant_fund_agent.modeling import service

    data_overrides = data_overrides or {}
    settings = get_settings().with_data_overrides(**data_overrides)
    dir_ = data_dir or settings.data.data_dir or os.getenv("DATA_DIR", "ticker_data")
    if data_dir or settings.data.provider == "lobster":
        os.environ["DATA_DIR"] = dir_
        service.DATA_DIR = dir_
    # An explicit ticker list takes precedence over the count cap (and the latter
    # shared-with-the-architect env var only matters if the modeling cache is cold).
    if tickers is None and n_tickers is not None:
        os.environ["ARCHITECT_N_TICKERS"] = str(n_tickers)
    if tickers:
        os.environ["QF_DATA_TICKERS"] = ",".join(tickers)

    discover_factors()

    log.info(
        "loading comparison panel: provider=%s asset_class=%s frequency=%s "
        "universe=%s start=%s end=%s",
        settings.data.provider, settings.data.asset_class, settings.data.frequency,
        settings.data.universe_preset or ("explicit" if tickers else "auto"),
        settings.data.start, settings.data.end,
    )
    # Provider-aware load: LOBSTER still routes to its CSV provider, while API
    # providers (yfinance/FMP/AlphaVantage) resolve universe + date range from the
    # active settings / explicit overrides.
    panel = load_panel(
        dir_ if settings.data.provider == "lobster" else None,
        tickers=tickers,
        n_tickers=None if tickers else n_tickers,
        settings=settings,
    )
    if tickers:
        loaded = list(next(iter(panel.values())).columns) if panel else []
        missing = [t for t in tickers if t not in loaded]
        log.info("explicit universe: %d/%d requested tickers loaded%s",
                 len(loaded), len(tickers),
                 f" (missing: {missing})" if missing else "")
    panel = _restrict_to_periods(panel, is_window, oos_window)
    panel = _subsample_bars(panel, max_bars)
    service._PANEL_CACHE = panel
    service._SIGNAL_CACHE = {}   # flush stale signals from any prior run
    return panel


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
    n = len(factor_ids)
    log.info("usable_factor_ids: probing %d factors (computing signals) ...", n)
    for i, fid in enumerate(factor_ids):
        if i % 20 == 0:
            log.info("usable_factor_ids: [%d/%d] ...", i + 1, n)
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
