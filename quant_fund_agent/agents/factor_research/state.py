"""State for the Factor Researcher Agent."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from quant_fund_agent.schemas import BacktestMetrics


class PaperSnippet(BaseModel):
    """Short, LLM-ready summary of a paper that's been selected for reading."""
    paper_id: str
    title: str
    published_date: date | None = None
    text: str = ""        # truncated extracted body
    file_path: str = ""


class FactorIdea(BaseModel):
    """The LLM's brainstorm output for a single proposed factor.

    Crucially we capture the ``trading_idea`` (the *why*) separately
    from ``code`` (the *how*) so it can be stored on the FactorRecord —
    every researcher factor carries its own thesis the same way seed
    factors carry their ``description``.
    """
    factor_id: str
    name: str
    category: str = "other"
    trading_idea: str = ""
    description: str = ""
    source_paper_ids: list[str] = Field(default_factory=list)
    code: str = ""


class FactorCandidate(BaseModel):
    """An idea that has been materialised into runnable code and backtested."""
    idea: FactorIdea
    code_path: str = ""
    backtest_metrics: BacktestMetrics | None = None
    rejected_reason: str = ""


class FactorResearcherState(BaseModel):
    """State that flows through the Factor Researcher Agent graph."""

    # ── inputs ──
    session_id: str = ""
    cutoff_date: date | None = None
    n_papers: int = 2
    n_factor_ideas: int = 3
    ic_threshold: float = 0.01
    ic_target_horizon: int = 6  # 1-minute bars at 10-sec resolution
    paper_sample_strategy: str = "unread_first"  # or "random"
    max_chars_per_paper: int = 30_000

    # ── data path (where load_panel reads from) ──
    data_dir: str = "ticker_data"

    # Cap the universe loaded for the research-time IC backtest.
    # ``None`` means "load every ticker in ``data_dir``"; useful values
    # are 10–20 on memory-constrained machines.  This only affects the
    # research session — the production 1-year backtest is unchanged.
    n_tickers: int | None = 15

    # ── populated by load_papers ──
    selected_papers: list[PaperSnippet] = Field(default_factory=list)

    # ── populated by brainstorm ──
    factor_ideas: list[FactorIdea] = Field(default_factory=list)

    # ── populated by materialise + backtest ──
    candidates: list[FactorCandidate] = Field(default_factory=list)

    # ── populated by filter_and_persist ──
    kept_factor_ids: list[str] = Field(default_factory=list)
    rejected_factor_ids: list[str] = Field(default_factory=list)

    # ── audit ──
    notes: list[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
