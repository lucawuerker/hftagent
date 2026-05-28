"""Pydantic models for factors, papers, strategies, and shared types."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TradingIdeaCategory(str, Enum):
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    CARRY = "carry"
    VALUE = "value"
    VOLATILITY = "volatility"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    SENTIMENT = "sentiment"
    MICROSTRUCTURE = "microstructure"
    OTHER = "other"


class FactorStatus(str, Enum):
    CANDIDATE = "candidate"
    BACKTESTED = "backtested"
    APPROVED = "approved"
    REJECTED = "rejected"


class FactorSource(str, Enum):
    """Where a factor came from.

    - ``SEED`` factors are the hard-coded baseline (e.g. the WorldQuant
      alphas). They are present at simulation start and survive across
      runs.
    - ``RESEARCHER`` factors are produced by the Factor Researcher Agent
      during a research session. They are *not* part of the baseline
      ensemble — they are scoped to the session/simulation in which the
      agent invented them and can be purged cleanly via
      ``FactorDatabase.purge_researcher_factors``.
    """
    SEED = "seed"
    RESEARCHER = "researcher"


class PaperStatus(str, Enum):
    UNREAD = "unread"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class StrategyStatus(str, Enum):
    DRAFT = "draft"
    BACKTESTING = "backtesting"
    APPROVED = "approved"
    LIVE = "live"
    RETIRED = "retired"


class PMStatus(str, Enum):
    """Where a strategy sits in the portfolio manager's lifecycle.

    Independent of ``StrategyStatus`` (which tracks the *research* state):
    a strategy is APPROVED by the statistician first, then becomes
    NOT_DEPLOYED in the PM's universe.  The PM may then promote it to
    LIVE, PAUSE it, FLAG it for the research team to revisit, or RETIRE
    it permanently.
    """
    NOT_DEPLOYED = "not_deployed"
    LIVE = "live"
    PAUSED = "paused"
    FLAGGED_FOR_REVIEW = "flagged_for_review"
    RETIRED = "retired"


class PMMode(str, Enum):
    """How autonomous the Portfolio Manager Agent is."""
    SELECTOR = "selector"   # passive: PM picks strategies, construction is fixed
    ACTIVE = "active"       # PM also chooses the construction method / overrides


class PMPersonality(str, Enum):
    """Pre-baked risk personalities for the PM agent."""
    DEFENSIVE = "defensive"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class ConstructionMethod(str, Enum):
    """Portfolio construction algorithms exposed to the PM agent.

    All of these are implemented in ``quant_fund_agent.portfolio.construction``
    and produce a mapping ``{strategy_id: weight}`` summing to 1.
    """
    EQUAL_WEIGHT = "equal_weight"
    INVERSE_VOLATILITY = "inverse_volatility"
    MIN_VARIANCE = "min_variance"
    MEAN_VARIANCE = "mean_variance"
    MAX_SHARPE = "max_sharpe"
    RISK_PARITY = "risk_parity"
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"
    CUSTOM_LLM = "custom_llm"   # PM (active mode) picks weights directly


class VotingMethod(str, Enum):
    """How a PM committee aggregates several PMs' proposals.

    * ``SIMPLE_AVERAGE``  – the final weight for each strategy is the
      mean across every PM's proposal (PMs that didn't pick a strategy
      contribute 0 to its average).  Deterministic, no LLM.
    * ``WEIGHTED_AVERAGE`` – same, but each PM has an explicit weight.
      Useful for a "chair + committee" structure.
    * ``LLM_MODERATOR``   – one extra LLM call sees every proposal and
      its rationale and produces a synthesised allocation.  Use when
      PMs disagree on *which* strategies to deploy (not just sizing).
    """
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    LLM_MODERATOR = "llm_moderator"


# ---------------------------------------------------------------------------
# Trading idea
# ---------------------------------------------------------------------------

class TradingIdea(BaseModel):
    id: str
    name: str
    category: TradingIdeaCategory
    description: str = ""
    factor_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Factor / Feature
# ---------------------------------------------------------------------------

class BacktestMetrics(BaseModel):
    """Key metrics produced by a single-factor backtest."""
    information_coefficient: float | None = None
    ic_std: float | None = None
    ic_ir: float | None = None
    ic_hit_rate: float | None = None
    ic_by_horizon: dict[str, dict[str, float | int | None]] = Field(default_factory=dict)
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    turnover: float | None = None
    annualised_return: float | None = None
    hit_rate: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class FactorRecord(BaseModel):
    """Metadata record stored in the FactorDatabase.

    The ``class_name`` field links this record to the concrete
    ``BaseFactor`` subclass registered in the factor registry
    (e.g. ``"ThreeSoldiersSignal"``).

    ``source`` distinguishes the hard-coded *seed* factors that boot the
    simulation from *researcher* factors that the Factor Researcher
    Agent invents at runtime.  ``research_session_id`` tags a researcher
    factor with the session in which it was created (e.g. ``"2019-W03"``)
    so it can be tracked, audited, or selectively purged.
    """
    id: str
    name: str
    class_name: str = ""
    description: str = ""
    formula: str = ""
    trading_idea: str = ""
    trading_idea_ids: list[str] = Field(default_factory=list)
    category: TradingIdeaCategory = TradingIdeaCategory.OTHER
    status: FactorStatus = FactorStatus.CANDIDATE
    backtest_metrics: BacktestMetrics | None = None
    source_paper_ids: list[str] = Field(default_factory=list)
    source: FactorSource = FactorSource.SEED
    research_session_id: str = ""
    code_path: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Paper
# ---------------------------------------------------------------------------

class Paper(BaseModel):
    id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    published_date: date | None = None
    status: PaperStatus = PaperStatus.UNREAD
    key_ideas: list[str] = Field(default_factory=list)
    extracted_factor_ids: list[str] = Field(default_factory=list)
    file_path: str | None = None
    url: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Strategy backtest metrics
# ---------------------------------------------------------------------------

class StrategyBacktestMetrics(BaseModel):
    """Comprehensive metrics from a vectorised strategy backtest."""
    # return-based
    annualised_return: float | None = None
    annualised_volatility: float | None = None
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    calmar_ratio: float | None = None

    # drawdown
    max_drawdown: float | None = None
    max_drawdown_duration_bars: int | None = None

    # win / loss
    hit_rate: float | None = None
    profit_factor: float | None = None
    avg_win: float | None = None
    avg_loss: float | None = None

    # turnover & capacity
    avg_daily_turnover: float | None = None
    avg_positions_held: float | None = None

    # IC of the composite signal
    ic_mean: float | None = None
    ic_ir: float | None = None

    # cumulative PnL series stored externally; path or key goes here
    extra: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Strategy record
# ---------------------------------------------------------------------------

class StrategyRecord(BaseModel):
    """Metadata record stored in the StrategyDatabase.

    The ``class_name`` field links this record to the concrete
    ``BaseStrategy`` subclass registered in the strategy registry.

    The Portfolio Manager Agent reads the following PM-facing fields:

    - ``backtest_metrics`` — *in-sample* metrics produced by the architect.
    - ``oos_backtest_metrics`` — *out-of-sample* metrics from the statistician.
    - ``live_metrics`` — running metrics produced by the live/backtest harness.
    - ``correlations`` — pairwise correlation with every other strategy in
      the DB (keyed by other strategy_id).  Kept in sync by
      ``StrategyDatabase.refresh_correlations``.
    - ``returns_path`` — on-disk location of the persisted per-bar PnL
      series (CSV) used to compute correlations and rolling metrics.
    - ``pm_status`` — where the PM has placed the strategy in its
      lifecycle (NOT_DEPLOYED → LIVE → PAUSED / FLAGGED / RETIRED).
    - ``pm_notes`` / ``flagged_reason`` — free-form context for audit.
    """
    id: str
    name: str
    class_name: str = ""
    description: str = ""
    factor_ids: list[str] = Field(default_factory=list)
    trading_idea_ids: list[str] = Field(default_factory=list)
    status: StrategyStatus = StrategyStatus.DRAFT
    backtest_metrics: StrategyBacktestMetrics | None = None
    oos_backtest_metrics: StrategyBacktestMetrics | None = None
    live_metrics: StrategyBacktestMetrics | None = None
    model_type: str = ""
    model_params: dict[str, Any] = Field(default_factory=dict)
    holding_period: int = 1

    # ── PM-layer fields ──
    pm_status: PMStatus = PMStatus.NOT_DEPLOYED
    pm_notes: str = ""
    flagged_reason: str = ""
    correlations: dict[str, float] = Field(default_factory=dict)
    returns_path: str = ""

    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Portfolio records (PM agent output)
# ---------------------------------------------------------------------------

class PortfolioAllocation(BaseModel):
    """A single line of a portfolio: how much capital is on which strategy."""
    strategy_id: str
    weight: float = 0.0
    enabled: bool = True
    rationale: str = ""


class PortfolioRecord(BaseModel):
    """One snapshot of a Portfolio Manager's allocation decision.

    Every invocation of the PM agent produces a ``PortfolioRecord``.  These
    are append-only and form an audit trail of how capital allocation
    evolved over a backtest / simulation.

    A simulation can have multiple PMs running side-by-side (e.g. a
    defensive and an aggressive PM with different sleeves of capital);
    each produces its own stream of ``PortfolioRecord``s, distinguished by
    ``pm_name``.
    """
    id: str
    pm_name: str = "default_pm"
    mode: PMMode = PMMode.SELECTOR
    personality: PMPersonality = PMPersonality.BALANCED
    construction_method: ConstructionMethod = ConstructionMethod.EQUAL_WEIGHT
    target_n_strategies: int = 10

    allocations: list[PortfolioAllocation] = Field(default_factory=list)
    flagged_strategy_ids: list[str] = Field(default_factory=list)
    retired_strategy_ids: list[str] = Field(default_factory=list)

    # Expected portfolio-level numbers from the construction step
    # (Sharpe, vol, return — computed from the covariance matrix used).
    expected_metrics: dict[str, float] = Field(default_factory=dict)

    # Free-form context produced by the PM (LLM rationale in ACTIVE mode,
    # personality summary in SELECTOR mode).
    pm_rationale: str = ""

    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
