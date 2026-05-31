"""Pre-implemented modelling toolbox for the Architect Agent.

Exposes a small, stable API used by the MCP modeling server and its
in-process fallback:

- ``list_model_menu`` / ``MODEL_SPECS`` / ``build_estimator`` — the catalog of
  fittable models (regression, tree ensembles, gradient boosting).
- ``fit_and_backtest`` — fit a chosen model on the in-sample window and
  backtest it (or back-test the static-weights baseline).
- feature helpers for turning the factor panel into a supervised dataset.
"""

from __future__ import annotations

from quant_fund_agent.modeling.catalog import (
    FITTABLE_MODEL_TYPES,
    MODEL_SPECS,
    STATIC_WEIGHTS,
    available_model_types,
    build_estimator,
    get_model_spec,
    is_model_available,
    list_model_menu,
    prepare_params,
)
from quant_fund_agent.modeling.features import (
    build_training_matrix,
    predict_to_signal,
    stack_features,
)
from quant_fund_agent.modeling.train import fit_and_backtest

__all__ = [
    "MODEL_SPECS",
    "FITTABLE_MODEL_TYPES",
    "STATIC_WEIGHTS",
    "available_model_types",
    "is_model_available",
    "build_estimator",
    "get_model_spec",
    "list_model_menu",
    "prepare_params",
    "stack_features",
    "build_training_matrix",
    "predict_to_signal",
    "fit_and_backtest",
]
