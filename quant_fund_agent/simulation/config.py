"""Configuration for a walk-forward backtest.

Everything that parameterises a run lives here in one serialisable dataclass so
a run is fully reproducible (the config is written to ``<output_dir>/config.json``).

The schedule model is deliberately simple and deterministic:

- A **weekly rebalance grid** (``grid_freq``) anchored at ``start``.
- No meetings fire until ``warmup`` of history exists (the first *cutoff* is the
  first grid point with ``cutoff - start >= warmup``) — so the agents always have
  enough data to research and fit on.
- Each meeting type (research / strategy / PM) has its own **cadence**, expressed
  as a multiple of the grid step (e.g. ``"1W"`` = every grid point, ``"1M"`` ≈ every
  4th).  At each grid point a meeting fires if it is due under its cadence.
- Between consecutive grid points the fund **trades** the current book bar-by-bar
  on the live (unseen) window.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from quant_fund_agent.schemas import (
    ConstructionMethod,
    PMMode,
    PMPersonality,
    VotingMethod,
)


class ExecutionModel(str, Enum):
    """How per-strategy positions are combined into the traded fund book.

    - ``NETTED`` (default): all live strategies' target positions, scaled by the
      PM's capital weights, are summed into ONE fund book per ticker — offsetting
      signals net out.  Fund-level ``max_positions`` / leverage / costs apply to
      the consolidated book.  The central-risk-book model.
    - ``POD``: each strategy trades its own dollar-neutral book and pays its own
      costs; the fund return is the capital-weighted sum of per-strategy returns.
      No cross-strategy netting (the multi-manager pod model).
    """

    NETTED = "netted"
    POD = "pod"


# ── offset parsing (e.g. "1W", "2W", "1M", "10D") ──────────────────────

_OFFSET_RE = re.compile(r"^\s*(\d+)\s*([DWM])\s*$", re.IGNORECASE)


def _parse_offset(spec: str) -> tuple[int, str]:
    m = _OFFSET_RE.match(spec or "")
    if not m:
        raise ValueError(
            f"Bad offset {spec!r}; use e.g. '1W', '2W', '1M', '10D'."
        )
    return int(m.group(1)), m.group(2).upper()


def offset_to_days(spec: str) -> int:
    """Approximate an offset spec as a number of *calendar* days.

    Months are treated as 30 days — fine for a thesis-scale warm-up window where
    only the rough amount of history matters, not the exact calendar.
    """
    n, unit = _parse_offset(spec)
    return n * {"D": 1, "W": 7, "M": 30}[unit]


def cadence_steps(cadence: str, grid_freq: str) -> int:
    """How many grid steps apart this cadence is (>= 1).

    ``cadence_steps("1M", "1W") == 4`` (a monthly meeting on a weekly grid).
    """
    grid_days = offset_to_days(grid_freq)
    cad_days = offset_to_days(cadence)
    return max(1, round(cad_days / grid_days))


@dataclass
class BacktestConfig:
    """All knobs for one walk-forward backtest run."""

    # ── horizon ──
    start: str                      # ISO date, inclusive
    end: str                        # ISO date, exclusive
    warmup: str = "1M"              # no meetings/trading until this much history
    grid_freq: str = "1W"          # the rebalance-grid step

    # ── meeting cadences (multiples of grid_freq) ──
    research_every: str = "1W"
    strategy_every: str = "1W"
    pm_rebalance_every: str = "1W"
    run_research: bool = True       # set False to skip factor-research meetings

    # ── how many strategies to research per meeting ──
    initial_strategies: int = 5     # at the FIRST strategy meeting (bootstrap)
    n_strategies_per_meeting: int = 2  # at every strategy meeting after

    # ── strategy pipeline params (threaded to the agents) ──
    max_iterations: int = 3
    oos_ratio: float = 0.2
    target_horizon: int = 6

    # ── execution / netting ──
    execution_model: ExecutionModel = ExecutionModel.NETTED
    fund_max_positions: int = 50        # fund-level cap (netted book)
    max_weight_per_name: float = 0.10   # per-ticker exposure cap (netted book)
    gross_leverage: float = 1.0         # cap on sum(|weights|) of the fund book
    capital: float = 10_000_000.0       # notional $ (PnL reporting multiplier)

    # ── transaction costs (spread-aware + commission, no market impact) ──
    use_spread_cost: bool = True
    spread_field: str = "effSpread"     # panel field used for the half-spread
    half_spread: bool = True            # charge 0.5 * spread (one side) per trade
    commission_bps: float = 0.5         # fixed commission per unit turnover
    # Flat synthetic spread (bps) applied only when ``spread_field`` is absent from
    # the panel (e.g. yfinance daily, which has no effSpread).  Default 0 keeps the
    # existing commission-only fallback; set e.g. 2.0 for a realistic large-cap
    # spread.  A high-low range proxy is deliberately NOT used — it overcharges by
    # ~100x vs real spreads.  Future: Corwin–Schultz high-low spread estimator.
    fallback_spread_bps: float = 0.0

    # ── PM configuration ──
    pm_mode: PMMode = PMMode.SELECTOR
    committee: bool = True
    pm_personalities: list[PMPersonality] = field(
        default_factory=lambda: [
            PMPersonality.DEFENSIVE,
            PMPersonality.BALANCED,
            PMPersonality.AGGRESSIVE,
        ]
    )
    voting_method: VotingMethod = VotingMethod.SIMPLE_AVERAGE
    construction_method: ConstructionMethod | None = None
    pm_name: str = "fund_committee"

    # ── factor universe (which factors strategies may be built from) ──
    # "all"     → the run's factor DB is seeded from the global permanent DB
    #             (seed factors + every researcher factor found by past preruns),
    #             and this run's research meetings append to it.
    # "session" → the run's factor DB is seeded with the SEED factors only, so the
    #             Selector sees seeds + only the factors discovered in THIS run's
    #             research meetings (permanent prerun factors are excluded).
    # Either way the run is scoped to its own DB under ``output_dir`` — a backtest
    # never writes the global ``data/factors/factor_db.json``.
    factor_universe: str = "all"    # "all" | "session"

    # ── named prerun selection (A/B different research LLMs) ──
    # When set, the run's factor DB is seeded from this named prerun's researcher
    # factors (data/factors/preruns/<prerun>/factor_db.json) instead of the global
    # library; ``factor_universe`` is then ignored.  ``include_seeds`` toggles
    # whether the 88 seed alphas are also visible to the Selector.  Typically used
    # with ``run_research=False`` to compare a prerun's factors downstream.
    prerun: str | None = None
    include_seeds: bool = True

    # ── data / universe ──
    data_dir: str = "ticker_data"
    n_tickers: int | None = None    # caps the universe via ARCHITECT_N_TICKERS

    # ── persistence ──
    run_id: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_root: str = "data/backtests"
    fresh: bool = True              # start from an empty strategy/portfolio book
    persist_every_n_steps: int = 4  # flush DBs to disk every N grid steps
    seed: int = 7

    # ── derived ──────────────────────────────────────────────────────────

    @property
    def start_date(self) -> date:
        return date.fromisoformat(self.start)

    @property
    def end_date(self) -> date:
        return date.fromisoformat(self.end)

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / self.run_id

    @property
    def strategy_db_path(self) -> Path:
        return self.output_dir / "strategy_db.json"

    @property
    def portfolio_db_path(self) -> Path:
        return self.output_dir / "portfolio_db.json"

    @property
    def returns_dir(self) -> Path:
        return self.output_dir / "returns"

    @property
    def factor_db_path(self) -> Path:
        """Run-scoped factor DB the research meetings write and the Selector reads.

        Pointed at by the ``FACTOR_DB_PATH`` env var for the duration of the run,
        so the global permanent factor DB is never touched by a backtest.
        """
        return self.output_dir / "factor_db.json"

    @property
    def factor_code_dir(self) -> Path:
        """Where this run's researcher ``.py`` files are snapshotted at the end.

        The generated code is recoverable from here even though it is purged from
        the package ``factors/researcher/`` dir so the package stays clean.
        """
        return self.output_dir / "factor_code"

    @property
    def paper_read_log_path(self) -> Path:
        """Run-scoped "papers already read" log (``PAPER_READ_LOG``).

        Starts empty on a fresh run so this backtest's paper selection is never
        biased by the prerun or by other runs, and the global paper index stays
        untouched.
        """
        return self.output_dir / "papers_read.json"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # enums → their values for clean JSON
        d["execution_model"] = self.execution_model.value
        d["pm_mode"] = self.pm_mode.value
        d["voting_method"] = self.voting_method.value
        d["pm_personalities"] = [p.value for p in self.pm_personalities]
        d["construction_method"] = (
            self.construction_method.value if self.construction_method else None
        )
        return d

    def __post_init__(self) -> None:
        # Coerce string enums (e.g. from a CLI) into the proper enum types.
        if isinstance(self.execution_model, str):
            self.execution_model = ExecutionModel(self.execution_model)
        if isinstance(self.pm_mode, str):
            self.pm_mode = PMMode(self.pm_mode)
        if isinstance(self.voting_method, str):
            self.voting_method = VotingMethod(self.voting_method)
        if isinstance(self.construction_method, str):
            self.construction_method = ConstructionMethod(self.construction_method)
        self.pm_personalities = [
            PMPersonality(p) if isinstance(p, str) else p
            for p in self.pm_personalities
        ]
        # Validate offset specs early (raises on bad input).
        for spec in (self.warmup, self.grid_freq, self.research_every,
                     self.strategy_every, self.pm_rebalance_every):
            _parse_offset(spec)
        if self.factor_universe not in ("all", "session"):
            raise ValueError(
                f"factor_universe must be 'all' or 'session', got "
                f"{self.factor_universe!r}."
            )
