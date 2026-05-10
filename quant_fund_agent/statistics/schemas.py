"""Pydantic schemas for statistical-test results."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StatTestVerdict(str, Enum):
    PASS = "pass"         # test is clearly passed
    WARN = "warn"         # borderline / worth flagging
    FAIL = "fail"         # test is clearly failed
    INFO = "info"         # purely informational, no pass/fail


class StatTestResult(BaseModel):
    """The standardised output of every statistical test.

    ``value`` is the main scalar the test produces (e.g. deflated SR,
    probability, p-value).  ``details`` carries any additional numbers
    the test wants to expose (e.g. all intermediate quantities).
    """
    test_id: str
    test_name: str
    verdict: StatTestVerdict = StatTestVerdict.INFO
    value: float | None = None
    threshold: float | None = None
    p_value: float | None = None
    summary: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
