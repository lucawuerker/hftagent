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
    assert validate_code(_GOOD, "hz_factor", expected_prediction_horizon=12) == "HzFactor"


def test_fixed_run_horizon_mismatch_rejected():
    with pytest.raises(CodeValidationError, match="fixed|6"):
        validate_code(_GOOD, "hz_factor", expected_prediction_horizon=6)


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


def test_deterministic_helper_code_allowed():
    code = _GOOD.replace(
        "import pandas as pd\n",
        """import pandas as pd
import numpy as np
from scipy import stats

def _paper_transform(x):
    # A paper-specific deterministic transform is fine when it is trailing-only.
    return np.log1p(x.abs()).rolling(3, min_periods=3).sum()

""",
    ).replace(
        "return data[\"close\"]",
        "close = data[\"close\"]\n        return _paper_transform(close).fillna(0.0)",
    )
    assert validate_code(code, "hz_factor") == "HzFactor"


@pytest.mark.parametrize(
    "body",
    [
        'return data["close"].shift(-1)',
        'return data["close"].diff(periods=-1)',
        'return data["close"].pct_change(-1)',
        'return data["close"].rolling(5, center=True).mean()',
    ],
)
def test_temporal_leakage_patterns_rejected(body):
    code = _GOOD.replace("return data[\"close\"]", body)
    with pytest.raises(CodeValidationError, match="look-ahead"):
        validate_code(code, "hz_factor")


def test_full_panel_fit_rejected():
    code = _GOOD.replace(
        "import pandas as pd\n",
        "import pandas as pd\nfrom sklearn.linear_model import LinearRegression\n",
    ).replace(
        "return data[\"close\"]",
        "model = LinearRegression().fit(data[\"close\"].fillna(0.0), data[\"close\"].fillna(0.0))\n"
        "        return data[\"close\"]",
    )
    with pytest.raises(CodeValidationError, match="fit"):
        validate_code(code, "hz_factor")
