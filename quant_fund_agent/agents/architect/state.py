"""State for the Architect Agent."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from quant_fund_agent.agents.selector.state import FactorSummary


class StrategySpec(BaseModel):
    """The strategy specification the LLM produces.

    Supports two strategy families:

    - ``static_weights`` (legacy): a hand-chosen weighted sum of factor
      signals, configured via ``weights``.
    - a fitted ML model (``ridge``, ``random_forest``, ``xgboost``, …): the
      model maps the ``factor_ids`` features to forward returns; the fitted
      estimator is persisted at ``model_artifact_path`` so the OOS test and the
      live backtest can reload it.
    """
    strategy_name: str = ""

    # ── model selection ──
    model_type: str = "static_weights"
    model_params: dict[str, Any] = Field(default_factory=dict)
    factor_ids: list[str] = Field(default_factory=list)   # ML features
    target_horizon: int = 6                               # bars ahead to predict
    model_artifact_path: str = ""                         # persisted fitted model

    # ── static-weights path ──
    weights: dict[str, float] = Field(default_factory=dict)

    # ── position construction ──
    holding_period: int = 6
    max_positions: int = 20
    equal_weight: bool = False
    min_conviction: float = 0.0

    # How the composite signal becomes positions — NOT chosen by the LLM; set
    # from the run's data type / override (see backtesting/positions.py) and
    # stamped here so OOS + live reproduce the same book.  "cross_sectional"
    # (dollar-neutral rank) or "per_underlying" (directional boundary, each
    # active name sized 1/max_positions).  ``position_params`` carries the
    # per_underlying knobs (mode/threshold/zscore_basis/zscore_window).
    position_construction: str = "cross_sectional"
    position_params: dict[str, Any] = Field(default_factory=dict)

    reasoning: str = ""


class TrialRecord(BaseModel):
    """One design-backtest iteration, kept for audit and deflated Sharpe."""
    iteration: int
    spec: StrategySpec
    metrics: dict[str, Any] = Field(default_factory=dict)


class ArchitectState(BaseModel):
    """State that flows through the architect agent graph."""

    # ── inputs (from the selector agent) ──
    hypothesis: str = ""
    selected_factor_ids: list[str] = Field(default_factory=list)
    factor_catalog: list[FactorSummary] = Field(default_factory=list)

    # ── train/test split ──
    # Fraction of the (chronological) panel reserved for the statistician's
    # out-of-sample test.  The architect only ever sees the in-sample slice.
    oos_split_ratio: float = 0.2

    # ── walk-forward cutoff (backtest only) ──
    # ISO date/datetime string.  When set, the architect only sees panel data
    # strictly before it (no look-ahead).  ``None`` → full history (production).
    as_of: str | None = None

    # ── prediction horizon (bars ahead the model forecasts; also the default
    #    holding period and the IC evaluation horizon).  Config-driven, not
    #    chosen by the LLM, so the showcase can sweep horizons cleanly. ──
    target_horizon: int = 6

    # ── position-construction regime for every strategy this run builds (resolved
    #    upstream from the data type / override; threaded onto each StrategySpec).
    position_construction: str = "cross_sectional"
    position_params: dict[str, Any] = Field(default_factory=dict)

    # ── current iteration ──
    strategy_spec: StrategySpec = Field(default_factory=StrategySpec)
    backtest_metrics: dict[str, Any] = Field(default_factory=dict)
    iteration: int = 0
    max_iterations: int = 5

    # ── history of all trials (for deflated Sharpe) ──
    trial_history: list[TrialRecord] = Field(default_factory=list)

    # ── decision after evaluate_and_decide ──
    decision: str = ""          # "approve" or "revise"
    decision_reasoning: str = ""

    # ── initial design reasoning (kept so the statistician / later stages
    #    can see both the original thesis and the final iteration's rationale) ──
    initial_reasoning: str = ""

    class Config:
        arbitrary_types_allowed = True
