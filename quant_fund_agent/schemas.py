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
    """
    id: str
    name: str
    class_name: str = ""
    description: str = ""
    formula: str = ""
    trading_idea_ids: list[str] = Field(default_factory=list)
    category: TradingIdeaCategory = TradingIdeaCategory.OTHER
    status: FactorStatus = FactorStatus.CANDIDATE
    backtest_metrics: BacktestMetrics | None = None
    source_paper_ids: list[str] = Field(default_factory=list)
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
    """
    id: str
    name: str
    class_name: str = ""
    description: str = ""
    factor_ids: list[str] = Field(default_factory=list)
    trading_idea_ids: list[str] = Field(default_factory=list)
    status: StrategyStatus = StrategyStatus.DRAFT
    backtest_metrics: StrategyBacktestMetrics | None = None
    model_type: str = ""
    model_params: dict[str, Any] = Field(default_factory=dict)
    holding_period: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
