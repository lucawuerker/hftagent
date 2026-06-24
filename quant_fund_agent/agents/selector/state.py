"""State for the Selector Agent."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FactorSummary(BaseModel):
    """Compact representation of a factor shown to the LLM."""
    factor_id: str
    name: str
    category: str
    description: str
    prediction_horizon: int = 6  # the factor's own forecast horizon, in bars
    ic_1: float | None = None
    ic_6: float | None = None
    ic_60: float | None = None
    icir_1: float | None = None
    icir_6: float | None = None
    icir_60: float | None = None


class SelectorState(BaseModel):
    """State that flows through the selector agent graph."""

    # populated by load_factor_catalog
    factor_catalog: list[FactorSummary] = Field(default_factory=list)

    # populated by formulate_hypothesis
    hypothesis: str = ""
    reasoning: str = ""

    # populated by select_factors
    selected_factor_ids: list[str] = Field(default_factory=list)
    selection_rationale: str = ""

    class Config:
        arbitrary_types_allowed = True
