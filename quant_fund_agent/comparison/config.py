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

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ic_horizons"] = list(self.ic_horizons)
        d["resolved_preruns"] = self.resolved_preruns()
        d["resolved_models"] = self.resolved_models()
        return d
