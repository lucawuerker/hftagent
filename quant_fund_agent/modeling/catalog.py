"""Catalog of pre-implemented models the Architect can fit.

This is the single source of truth for *which* models exist, their default
hyper-parameters, and the tunable ranges.  It drives three things at once:

1. the **menu** shown to the Architect LLM (so it knows what it can pick),
2. server-side **validation/clamping** of the LLM's chosen hyper-parameters
   (so a hallucinated ``n_estimators=10_000_000`` can't blow up the box), and
3. the actual **estimator construction** used by ``modeling.train``.

Estimators are scikit-learn-compatible regressors (``.fit`` / ``.predict``):
the model maps cross-sectionally-normalised factor signals → forward returns.
The prediction becomes the strategy's composite alpha signal, which then runs
through the *same* position-construction logic as every other strategy.

Heavy / optional imports (xgboost, lightgbm) are done lazily inside the
builders so that merely importing this module — e.g. to render the menu — never
requires those packages to be installed.
"""

from __future__ import annotations

import functools
import importlib
from dataclasses import dataclass, field
from typing import Any, Callable

# Fixed knobs we always inject (not exposed to the LLM as tunables).
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Parameter + model specs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamSpec:
    """One tunable hyper-parameter: type, default, and valid range/choices."""
    name: str
    kind: str                       # "float" | "int" | "bool" | "choice"
    default: Any
    min: float | None = None
    max: float | None = None
    choices: tuple[Any, ...] | None = None
    description: str = ""

    def coerce(self, value: Any) -> Any:
        """Coerce + clamp ``value`` into this spec's type and range.

        Returns the default for anything that can't be sensibly coerced —
        the LLM occasionally emits ``"0.1"`` (string), ``null``, or a wildly
        out-of-range number, and we never want that to crash a fit.
        """
        if value is None:
            return self.default
        try:
            if self.kind == "bool":
                if isinstance(value, str):
                    return value.strip().lower() in ("true", "1", "yes")
                return bool(value)
            if self.kind == "int":
                v = int(round(float(value)))
                if self.min is not None:
                    v = max(int(self.min), v)
                if self.max is not None:
                    v = min(int(self.max), v)
                return v
            if self.kind == "float":
                v = float(value)
                if self.min is not None:
                    v = max(self.min, v)
                if self.max is not None:
                    v = min(self.max, v)
                return v
            if self.kind == "choice":
                return value if (self.choices and value in self.choices) else self.default
        except (TypeError, ValueError):
            return self.default
        return self.default

    def to_menu(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "type": self.kind,
            "default": self.default,
            "description": self.description,
        }
        if self.choices is not None:
            d["choices"] = list(self.choices)
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        return d


@dataclass(frozen=True)
class ModelSpec:
    """A fittable model: how to describe it, validate it, and build it."""
    model_type: str
    family: str                     # "baseline" | "linear" | "tree" | "boosting"
    description: str
    builder: Callable[[dict[str, Any]], Any]
    params: tuple[ParamSpec, ...] = field(default_factory=tuple)

    def prepare_params(self, raw: dict[str, Any] | None) -> dict[str, Any]:
        """Merge defaults, drop unknowns, coerce + clamp every value."""
        raw = raw or {}
        return {p.name: p.coerce(raw.get(p.name, p.default)) for p in self.params}

    def build(self, raw: dict[str, Any] | None):
        return self.builder(self.prepare_params(raw))

    def to_menu(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "family": self.family,
            "description": self.description,
            "params": [p.to_menu() for p in self.params],
        }


# ---------------------------------------------------------------------------
# Builders (lazy imports so optional deps stay optional)
# ---------------------------------------------------------------------------

def _build_linear(p: dict[str, Any]):
    from sklearn.linear_model import LinearRegression
    return LinearRegression(fit_intercept=p["fit_intercept"])


def _build_ridge(p: dict[str, Any]):
    from sklearn.linear_model import Ridge
    return Ridge(alpha=p["alpha"], random_state=RANDOM_STATE)


def _build_lasso(p: dict[str, Any]):
    from sklearn.linear_model import Lasso
    return Lasso(alpha=p["alpha"], random_state=RANDOM_STATE, max_iter=5000)


def _build_elastic_net(p: dict[str, Any]):
    from sklearn.linear_model import ElasticNet
    return ElasticNet(
        alpha=p["alpha"], l1_ratio=p["l1_ratio"],
        random_state=RANDOM_STATE, max_iter=5000,
    )


def _build_random_forest(p: dict[str, Any]):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"] or None,
        min_samples_leaf=p["min_samples_leaf"],
        max_features=p["max_features"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _build_gradient_boosting(p: dict[str, Any]):
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        learning_rate=p["learning_rate"],
        subsample=p["subsample"],
        random_state=RANDOM_STATE,
    )


def _build_xgboost(p: dict[str, Any]):
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        learning_rate=p["learning_rate"],
        subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        reg_lambda=p["reg_lambda"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )


def _build_lightgbm(p: dict[str, Any]):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=p["n_estimators"],
        num_leaves=p["num_leaves"],
        max_depth=p["max_depth"],
        learning_rate=p["learning_rate"],
        subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------

MODEL_SPECS: dict[str, ModelSpec] = {
    "linear_regression": ModelSpec(
        model_type="linear_regression",
        family="linear",
        description="Ordinary least squares — the linear baseline above static weights.",
        builder=_build_linear,
        params=(
            ParamSpec("fit_intercept", "bool", True,
                      description="Whether to fit an intercept term."),
        ),
    ),
    "ridge": ModelSpec(
        model_type="ridge",
        family="linear",
        description="L2-regularised linear regression; robust to collinear factors.",
        builder=_build_ridge,
        params=(
            ParamSpec("alpha", "float", 1.0, min=1e-4, max=1000.0,
                      description="L2 penalty strength; higher = more shrinkage."),
        ),
    ),
    "lasso": ModelSpec(
        model_type="lasso",
        family="linear",
        description="L1-regularised linear regression; performs factor selection.",
        builder=_build_lasso,
        params=(
            ParamSpec("alpha", "float", 1e-3, min=1e-5, max=1.0,
                      description="L1 penalty strength; higher = sparser model."),
        ),
    ),
    "elastic_net": ModelSpec(
        model_type="elastic_net",
        family="linear",
        description="Mix of L1 and L2 regularisation (selection + shrinkage).",
        builder=_build_elastic_net,
        params=(
            ParamSpec("alpha", "float", 1e-3, min=1e-5, max=1.0,
                      description="Overall penalty strength."),
            ParamSpec("l1_ratio", "float", 0.5, min=0.0, max=1.0,
                      description="0 = pure ridge, 1 = pure lasso."),
        ),
    ),
    "random_forest": ModelSpec(
        model_type="random_forest",
        family="tree",
        description="Bagged decision trees; captures non-linear factor interactions.",
        builder=_build_random_forest,
        params=(
            ParamSpec("n_estimators", "int", 200, min=10, max=1000,
                      description="Number of trees."),
            ParamSpec("max_depth", "int", 6, min=0, max=40,
                      description="Max tree depth (0 = unlimited)."),
            ParamSpec("min_samples_leaf", "int", 50, min=1, max=5000,
                      description="Min samples per leaf; higher = more regularised."),
            ParamSpec("max_features", "float", 0.5, min=0.1, max=1.0,
                      description="Fraction of features considered per split."),
        ),
    ),
    "gradient_boosting": ModelSpec(
        model_type="gradient_boosting",
        family="boosting",
        description="Sequentially boosted shallow trees (scikit-learn).",
        builder=_build_gradient_boosting,
        params=(
            ParamSpec("n_estimators", "int", 200, min=10, max=2000,
                      description="Number of boosting stages."),
            ParamSpec("max_depth", "int", 3, min=1, max=12,
                      description="Depth of each tree."),
            ParamSpec("learning_rate", "float", 0.05, min=1e-3, max=0.5,
                      description="Shrinkage applied to each tree."),
            ParamSpec("subsample", "float", 0.8, min=0.3, max=1.0,
                      description="Row subsampling fraction per stage."),
        ),
    ),
    "xgboost": ModelSpec(
        model_type="xgboost",
        family="boosting",
        description="XGBoost gradient boosting — a quant-ML workhorse.",
        builder=_build_xgboost,
        params=(
            ParamSpec("n_estimators", "int", 300, min=10, max=2000,
                      description="Number of boosting rounds."),
            ParamSpec("max_depth", "int", 4, min=1, max=12,
                      description="Depth of each tree."),
            ParamSpec("learning_rate", "float", 0.05, min=1e-3, max=0.5,
                      description="Step-size shrinkage."),
            ParamSpec("subsample", "float", 0.8, min=0.3, max=1.0,
                      description="Row subsampling fraction."),
            ParamSpec("colsample_bytree", "float", 0.8, min=0.3, max=1.0,
                      description="Feature subsampling fraction per tree."),
            ParamSpec("reg_lambda", "float", 1.0, min=0.0, max=100.0,
                      description="L2 regularisation on leaf weights."),
        ),
    ),
    "lightgbm": ModelSpec(
        model_type="lightgbm",
        family="boosting",
        description="LightGBM gradient boosting — fast, leaf-wise trees.",
        builder=_build_lightgbm,
        params=(
            ParamSpec("n_estimators", "int", 300, min=10, max=2000,
                      description="Number of boosting rounds."),
            ParamSpec("num_leaves", "int", 31, min=7, max=255,
                      description="Max leaves per tree; the main capacity knob."),
            ParamSpec("max_depth", "int", -1, min=-1, max=20,
                      description="Max tree depth (-1 = unlimited)."),
            ParamSpec("learning_rate", "float", 0.05, min=1e-3, max=0.5,
                      description="Step-size shrinkage."),
            ParamSpec("subsample", "float", 0.8, min=0.3, max=1.0,
                      description="Row subsampling fraction."),
            ParamSpec("colsample_bytree", "float", 0.8, min=0.3, max=1.0,
                      description="Feature subsampling fraction per tree."),
        ),
    ),
}


# Models that are actually fit from data (everything except the static baseline).
FITTABLE_MODEL_TYPES: tuple[str, ...] = tuple(MODEL_SPECS.keys())

# The legacy non-fitted strategy: a static weighted sum of factor signals.
STATIC_WEIGHTS = "static_weights"


# Models backed by a native library that may be absent (needs an OpenMP runtime
# such as libomp on macOS).  Everything else is pure scikit-learn (a hard dep).
_NATIVE_MODULE: dict[str, str] = {"xgboost": "xgboost", "lightgbm": "lightgbm"}


@functools.lru_cache(maxsize=None)
def is_model_available(model_type: str) -> bool:
    """Whether ``model_type`` can actually be built in this environment.

    scikit-learn models and the static baseline are always available; xgboost /
    lightgbm are probed by importing the package (their native lib loads on
    import), so a missing ``libomp`` cleanly removes them from the menu instead
    of crashing a fit.
    """
    if model_type == STATIC_WEIGHTS:
        return True
    if model_type not in MODEL_SPECS:
        return False
    module = _NATIVE_MODULE.get(model_type)
    if module is None:
        return True
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def available_model_types(include_static: bool = True) -> list[str]:
    """The model types that are usable right now (unavailable natives dropped)."""
    out = [m for m in MODEL_SPECS if is_model_available(m)]
    return [STATIC_WEIGHTS, *out] if include_static else out


def get_model_spec(model_type: str) -> ModelSpec:
    if model_type not in MODEL_SPECS:
        raise KeyError(
            f"Unknown model_type {model_type!r}. "
            f"Available: {sorted(MODEL_SPECS)} (+ {STATIC_WEIGHTS!r})."
        )
    return MODEL_SPECS[model_type]


def build_estimator(model_type: str, params: dict[str, Any] | None):
    """Construct a fresh estimator with validated + clamped hyper-parameters."""
    return get_model_spec(model_type).build(params)


def prepare_params(model_type: str, params: dict[str, Any] | None) -> dict[str, Any]:
    """Return the cleaned hyper-parameter dict that would be used to build."""
    return get_model_spec(model_type).prepare_params(params)


def list_model_menu() -> list[dict[str, Any]]:
    """The full model menu (incl. the static baseline) for the Architect LLM."""
    menu = [
        {
            "model_type": STATIC_WEIGHTS,
            "family": "baseline",
            "description": (
                "Hand-chosen static weighted sum of factor signals (no fitting). "
                "Provide a `weights` map instead of `model_params`."
            ),
            "params": [],
        }
    ]
    menu.extend(
        spec.to_menu() for spec in MODEL_SPECS.values()
        if is_model_available(spec.model_type)
    )
    return menu
