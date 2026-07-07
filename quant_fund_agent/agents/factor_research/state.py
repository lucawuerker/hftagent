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
    source: str = ""      # "local_pdf" | "link" | "abstract"


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
    # The forward offset (in *bars*) at which this factor's edge is expected to
    # peak, and optional alternatives worth measuring.  Chosen by the brainstorm
    # LLM (told the feed's bar size); the materialised class attribute is the
    # source of truth, this seeds the codegen spec.
    prediction_horizon: int = 6
    suggested_horizons: list[int] = Field(default_factory=list)
    # Falsifiable direction claim (+1/-1) from the Hypothesis agent (evolution
    # engine, P3+); feeds the harness's sign-consistency check.  ``None`` (the
    # oneshot brainstorm) skips that check.
    expected_sign: int | None = None
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
    # Total idea budget for the session.  Brainstorm reads one paper per
    # LLM call and splits this budget across the selected papers, then
    # dedupes + trims the pooled ideas back down to this many.
    n_factor_ideas: int = 3
    # Horizon (in 10-sec bars) at which each factor's IC is *measured and
    # recorded* for reference.  It is no longer an accept/reject gate — every
    # successfully-backtested factor is kept regardless of IC magnitude.
    ic_target_horizon: int = 6  # 1-minute bars at 10-sec resolution
    # When True, brainstorm/codegen must use ``ic_target_horizon`` as the factor's
    # declared ``prediction_horizon`` instead of choosing per idea.
    force_prediction_horizon: bool = False
    # Bar size (seconds per bar) of the configured feed, inferred once from the
    # panel index (see ``data.frequency``).  Surfaced to the brainstorm/codegen
    # prompts so the LLM can reason about prediction horizons in wall-clock time.
    # ``None`` → feed-agnostic prompt wording (no assumed default).
    seconds_per_bar: float | None = None
    paper_sample_strategy: str = "unread_first"  # or "random"
    # What new factor ids are de-duped against during brainstorm:
    #   "package" → every registered factor class (all code in the shared
    #               researcher package) ∪ the active factor DB (default).
    #   "prerun"  → only the active prerun's own factor DB, so a prerun's
    #               brainstorm is not anchored by other preruns' factors (the
    #               materialise step still drops any id that would clash with an
    #               existing class, so the shared package is never corrupted).
    dedup_scope: str = "package"  # "package" | "prerun"
    # Per-paper character budget for the text fed to the LLM.  Set high
    # enough to include the *entire* body of a normal academic paper
    # (methodology, results, conclusions) — not just the abstract/intro —
    # so the researcher reads the whole paper.  The cap is only a safety
    # bound against a pathological multi-hundred-page PDF blowing the LLM
    # context window; a typical 30–40 page paper fits well within it.
    max_chars_per_paper: int = 200_000

    # ── data path (where load_panel reads from) ──
    data_dir: str = "ticker_data"

    # Fields the configured data feed can actually supply this run (provider
    # capabilities + LOBSTER order-book level + the fundamentals opt-out; see
    # ``quant_fund_agent.data.usable_fields``).  The brainstorm/codegen DATA
    # CONTEXT is built from this set and produced factors are gated against it,
    # so the researcher only invents factors this run can serve.  ``None`` →
    # un-gated (every LOBSTER field + fundamentals), the historical behaviour.
    allowed_fields: list[str] | None = None

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
