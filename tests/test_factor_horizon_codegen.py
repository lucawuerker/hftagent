"""Codegen validation of the factor ``prediction_horizon`` class attribute.

The researcher's static validator (mirroring the ``inputs`` check) requires every
generated factor to declare a positive-int ``prediction_horizon`` and, if present,
a list of positive-int ``suggested_horizons``.  These are pure static-AST checks —
no registry, no smoke test.
"""

from __future__ import annotations

import pytest

from quant_fund_agent.agents.factor_research.codegen import (
    CodeValidationError,
    validate_code,
)
from quant_fund_agent.agents.factor_research.prompts import MAX_PREDICTION_HORIZON

_GOOD = '''
from __future__ import annotations
import pandas as pd
from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor

@register_factor
class HzFactor(BaseFactor):
    factor_id = "hz_factor"
    category = "momentum"
    inputs = ["close"]
    prediction_horizon = 12
    suggested_horizons = [1, 12, 60]
    def calc(self, data):
        return data["close"]
'''


def test_valid_horizon_passes():
    assert validate_code(_GOOD, "hz_factor") == "HzFactor"


def test_missing_horizon_rejected():
    code = _GOOD.replace("    prediction_horizon = 12\n", "")
    with pytest.raises(CodeValidationError, match="prediction_horizon"):
        validate_code(code, "hz_factor")


@pytest.mark.parametrize("bad", ["0", "-3"])
def test_non_positive_horizon_rejected(bad):
    code = _GOOD.replace("prediction_horizon = 12", f"prediction_horizon = {bad}")
    with pytest.raises(CodeValidationError, match="positive"):
        validate_code(code, "hz_factor")


def test_non_int_horizon_rejected():
    code = _GOOD.replace("prediction_horizon = 12", "prediction_horizon = 'soon'")
    with pytest.raises(CodeValidationError, match="prediction_horizon"):
        validate_code(code, "hz_factor")


def test_over_cap_horizon_rejected():
    code = _GOOD.replace("prediction_horizon = 12",
                         f"prediction_horizon = {MAX_PREDICTION_HORIZON + 1}")
    with pytest.raises(CodeValidationError, match="positive|≤"):
        validate_code(code, "hz_factor")


def test_bad_suggested_horizon_rejected():
    code = _GOOD.replace("suggested_horizons = [1, 12, 60]",
                         "suggested_horizons = [1, 0, 60]")
    with pytest.raises(CodeValidationError, match="suggested_horizons"):
        validate_code(code, "hz_factor")


def test_suggested_horizons_optional():
    code = _GOOD.replace("    suggested_horizons = [1, 12, 60]\n", "")
    assert validate_code(code, "hz_factor") == "HzFactor"
