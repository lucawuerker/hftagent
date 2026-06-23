"""Configuration for one research-LLM comparison run.

Everything that parameterises a comparison lives here in one serialisable
dataclass, written to ``<output_dir>/config.json`` for reproducibility.  The
*only* thing that varies between the preruns being compared is their factor set
(produced by different research models) — the panel, IS/OOS split, horizon and
model hyper-parameters are held identical, which is the fair-comparison invariant.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("comparison.config")

# Forward-return horizons (in bars) for the single-factor IC track.  On the
# 10-second LOBSTER panel these are 10s / ~1m / ~10m — matching the user's
# prelim notebook (data/factors/factor_db.json::ic_by_horizon keys "1"/"6"/"60").
DEFAULT_IC_HORIZONS: tuple[int, ...] = (1, 6, 60)

# Lighter hyper-parameters swapped in under ``--fast`` (merged over catalog
# defaults; unknown keys are dropped per model, so it is safe to apply to any
# model_type — linear models simply ignore ``n_estimators``).
_FAST_MODEL_PARAMS: dict[str, dict[str, object]] = {
    "random_forest": {"n_estimators": 60},
    "gradient_boosting": {"n_estimators": 60},
    "xgboost": {"n_estimators": 80},
    "lightgbm": {"n_estimators": 120},
}


def _period_bound(token: str, *, end: bool):
    """Parse a ``YYYY-MM`` or ``YYYY-MM-DD`` token into a boundary timestamp.

    ``end=False`` → first instant of that month/day; ``end=True`` → last instant
    (so a whole-month token expands to the entire month, a whole-day token to the
    entire day — both inclusive of every intraday bar within).
    """
    import pandas as pd

    token = token.strip()
    parts = token.split("-")
    if len(parts) == 2:  # YYYY-MM
        start = pd.Timestamp(f"{token}-01")
        if not end:
            return start
        return start + pd.offsets.MonthBegin(1) - pd.Timedelta(nanoseconds=1)
    if len(parts) == 3:  # YYYY-MM-DD
        day = pd.Timestamp(token)
        return day if not end else day + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    raise ValueError(f"Bad date token {token!r}; use YYYY-MM or YYYY-MM-DD.")


def period_mask(spec: str, index) -> Any:
    """Boolean numpy mask over a ``DatetimeIndex`` for a calendar period spec.

    ``spec`` is either a comma-separated list of months/dates
    (``"2024-06,2024-07"`` / ``"2024-06-15"``) or an inclusive range
    ``START:END`` where each end is a month or date (a month at the END of a range
    expands to the whole month, e.g. ``"2024-06:2024-08"`` covers all of August).
    """
    import numpy as np

    spec = spec.strip()
    if ":" in spec:
        a, b = spec.split(":", 1)
        lo, hi = _period_bound(a, end=False), _period_bound(b, end=True)
        return np.asarray((index >= lo) & (index <= hi))
    mask = np.zeros(len(index), dtype=bool)
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        lo, hi = _period_bound(tok, end=False), _period_bound(tok, end=True)
        mask |= np.asarray((index >= lo) & (index <= hi))
    return mask


@dataclass
class ComparisonConfig:
    """All knobs for one comparison of several preruns' factor sets."""

    # ── which preruns to compare (empty → auto-discover every prerun on disk) ──
    preruns: list[str] = field(default_factory=list)

    # ── brute-force model coverage (None → every available catalog model) ──
    models: list[str] | None = None
    include_ensemble: bool = True

    # ── which factor universe each prerun contributes ──
    # False (default) → compare only the *researched* factors (the clean A/B of
    # research models); True → also expose the 88 seed alphas to every track.
    include_seeds: bool = False

    # ── which tracks to run ──
    run_ic: bool = True
    run_analytics: bool = True       # diversity/redundancy + deflation/importance
    run_bruteforce: bool = True
    run_downstream: bool = True

    # ── shared evaluation params ──
    ic_horizons: tuple[int, ...] = DEFAULT_IC_HORIZONS
    target_horizon: int = 6           # brute-force / downstream forecast horizon
    oos_split_ratio: float = 0.2      # held-out tail fraction
    holding_period: int | None = None  # None → target_horizon
    max_positions: int = 20

    # ── speed knobs (brute-force track) ──
    # train_sample_frac < 1.0 fits each model on a random subset of the training
    # rows (the backtest still uses all data); the dominant lever for the heavy
    # tree/boosting models on a large panel.  `fast` is a convenience preset that
    # also swaps in lighter tree/boosting hyper-parameters.  Pair either with
    # `n_tickers` (fewer underlyings → a smaller panel speeds up *every* model
    # and the IC track too).
    fast: bool = False
    train_sample_frac: float = 1.0

    # ── ML-combined-signal vectorised backtest (brute-force track) ──
    # Each catalog model fits the factors (standardised per-underlying over time on
    # the IS window — `fit_standardize`) and predicts ONE combined signal per
    # (timestamp, underlying).  That signal is run through a *simple* per-underlying
    # vectorised backtest: position(signal) × the underlying's own forward return,
    # then aggregated.  Every modelling choice below is a changeable argument.
    # `fit_scope` decides whether a model is fit ONCE across all underlyings'
    # pooled rows ("pooled", the default — suits homogeneous, data-light universes
    # like S&P100 stocks) or SEPARATELY per underlying ("per_underlying" — a model
    # per name, suits heterogeneous, data-rich ones like the LOBSTER ETFs).
    fit_scope: str = "pooled"                 # "pooled" | "per_underlying"
    fit_standardize: str = "per_underlying"   # "per_underlying" | "cross_sectional"
    position_mode: str = "threshold"          # "threshold" | "sign" | "continuous"
    position_threshold: float = 1.0           # ±t (in z units) for the threshold band
    position_zscore_basis: str = "expanding"  # "expanding" | "full" | "rolling" | "none"
    position_zscore_window: int = 500         # window for the "rolling" basis
    backtest_aggregation: str = "portfolio"   # "portfolio" | "per_underlying"

    # ── downstream-agent track ──
    n_strategies: int = 3             # strategies built per prerun
    committee: bool = True            # PM committee vs single PM

    # ── data / universe ──
    data_dir: str = field(default_factory=lambda: os.getenv("DATA_DIR", "ticker_data"))
    n_tickers: int | None = None      # cap universe via ARCHITECT_N_TICKERS (None = all)
    # Explicit universe — an exact ticker list (e.g. ["AAPL", "MSFT", "CORN"]).  Takes
    # precedence over ``n_tickers`` when set; only these names are loaded.  None = all
    # tickers in the data dir (optionally capped by ``n_tickers``).
    tickers: list[str] | None = None
    # Calendar IS/OOS split.  When BOTH are set, the held-out split is by *date*
    # (these months/dates train, those test) instead of the ``oos_split_ratio`` tail
    # fraction, and the loaded panel is restricted to their union.  Each is a comma
    # list of months/dates ("2024-06,2024-07" / "2024-06-15") or an inclusive range
    # ("2024-06:2024-08").  Must be disjoint.  Default None → the ratio tail split.
    is_window: str | None = None      # train period spec
    oos_window: str | None = None     # OOS period spec
    # Exact chronological cutoff.  When set, IS is every bar strictly before this
    # timestamp and OOS is every bar at/after it.  This is less error-prone than
    # inclusive date windows for boundary dates such as 2024-06-01.
    split_date: str | None = None
    # Data-layer overrides.  None means "use quant.config.yaml / env".
    data_provider: str | None = None
    data_asset_class: str | None = None
    data_frequency: str | None = None
    data_start: str | None = None
    data_end: str | None = None
    data_universe_preset: str | None = None
    data_cache_dir: str | None = None
    # Uniform timestamp subsample of the whole panel (None = keep every bar).  The
    # single biggest speed lever on the intraday LOBSTER panel (~1.4M bars/ticker):
    # striding the index to `max_bars` slims *every* track at once (IC, analytics,
    # brute-force) and removes the brute-force OOM, since feature matrices are built
    # from the slim panel.  Horizons are then in retained-bar units (consistent
    # across preruns).  `--fast` defaults this to 20_000 when unset.
    max_bars: int | None = None

    # ── factor-analytics track (diversity/redundancy + deflation/importance) ──
    # Both reuse the already-cached factor signals, so they are near-free.
    analytics_max_rows: int = 50_000  # cap (timestamp×ticker) rows for corr / model fit
    corr_threshold: float = 0.7       # |corr| ≥ τ groups factors into a redundancy cluster
    importance_models: tuple[str, ...] = ("lasso", "gradient_boosting")

    # ── persistence ──
    checkpoint: bool = True           # persist tables+figures after EACH track (crash-safe)
    comparison_id: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_root: str = "data/comparisons"
    seed: int = 7

    # ── derived ───────────────────────────────────────────────────────────

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / self.comparison_id

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    def resolved_preruns(self) -> list[str]:
        """The preruns to compare — explicit list, or every prerun on disk."""
        from quant_fund_agent.factors import preruns

        if self.preruns:
            return list(self.preruns)
        return preruns.list_preruns()

    def resolved_models(self) -> list[str]:
        """Validated brute-force model list (drops unknown / unavailable types)."""
        from quant_fund_agent.modeling.catalog import (
            MODEL_SPECS,
            available_model_types,
            is_model_available,
        )

        if not self.models:
            return available_model_types(include_static=False)
        out: list[str] = []
        for m in self.models:
            if m not in MODEL_SPECS:
                log.warning("Unknown model_type %r — skipping (known: %s).",
                            m, sorted(MODEL_SPECS))
            elif not is_model_available(m):
                log.warning("model_type %r is not available in this env — skipping.", m)
            else:
                out.append(m)
        return out

    def fast_model_params(self, model_type: str) -> dict[str, Any] | None:
        """Lighter hyper-parameters for ``model_type`` under ``--fast`` (else None)."""
        if not self.fast:
            return None
        return _FAST_MODEL_PARAMS.get(model_type)

    def split_masks(self, index) -> tuple[Any, Any]:
        """``(is_mask, oos_mask)`` boolean arrays over a panel timestamp ``index``.

        Calendar mode (``is_window``/``oos_window`` both set) selects the IS/OOS
        rows by month/date and requires them disjoint; otherwise the default
        contiguous tail split by ``oos_split_ratio`` (first ``1-ratio`` = IS).
        """
        import numpy as np

        n = len(index)
        if self.split_date:
            if self.is_window or self.oos_window:
                raise ValueError(
                    "--split-date cannot be combined with --train-months/--oos-months."
                )
            split = datetime.fromisoformat(self.split_date)
            is_mask = np.asarray(index < split)
            oos_mask = np.asarray(index >= split)
            return is_mask, oos_mask
        if self.is_window or self.oos_window:
            if not (self.is_window and self.oos_window):
                raise ValueError(
                    f"Calendar IS/OOS split needs BOTH a train and an OOS window "
                    f"(got is_window={self.is_window!r}, oos_window={self.oos_window!r}).")
            is_mask = period_mask(self.is_window, index)
            oos_mask = period_mask(self.oos_window, index)
            overlap = int((is_mask & oos_mask).sum())
            if overlap:
                raise ValueError(
                    f"train and OOS windows overlap on {overlap} bars — they must be "
                    f"disjoint (is_window={self.is_window!r}, oos_window={self.oos_window!r}).")
            return is_mask, oos_mask
        cut = int(n * (1.0 - self.oos_split_ratio))
        is_mask = np.zeros(n, dtype=bool)
        is_mask[:cut] = True
        return is_mask, ~is_mask

    def window_union_mask(self, index) -> Any | None:
        """Boolean mask of the train∪OOS calendar windows (None if not in date mode)."""
        if self.split_date:
            return None
        if not (self.is_window and self.oos_window):
            return None
        return period_mask(self.is_window, index) | period_mask(self.oos_window, index)

    def data_overrides(self) -> dict[str, Any]:
        """DataSettings fields this comparison intentionally overrides."""
        mapping = {
            "provider": self.data_provider,
            "asset_class": self.data_asset_class,
            "frequency": self.data_frequency,
            "start": self.data_start,
            "end": self.data_end,
            "universe_preset": self.data_universe_preset,
            "cache_dir": self.data_cache_dir,
        }
        return {k: v for k, v in mapping.items() if v is not None}

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ic_horizons"] = list(self.ic_horizons)
        d["resolved_preruns"] = self.resolved_preruns()
        d["resolved_models"] = self.resolved_models()
        return d
