"""A strategy whose signal is the prediction of a fitted ML model.

``ModelStrategy`` is the runtime counterpart of :class:`DynamicStrategy`: where
``DynamicStrategy`` combines factor signals with static weights, ``ModelStrategy``
runs them through a fitted scikit-learn-style estimator (ridge, random forest,
XGBoost, …) whose output *is* the composite alpha signal.

It deliberately implements only ``calc`` — turning factor signals into a
``(time × ticker)`` prediction frame — so the existing backtester handles
position construction (z-score → conviction filter → top-N → dollar-neutral)
identically for static and ML strategies.  The only thing that differs between
the two is how the raw signal is produced.

The estimator is supplied either directly (right after the architect fits it) or
reloaded from a joblib artifact via :meth:`from_artifact` (the Statistician's
OOS test and the future live backtest).  It is **never refit** here.

Normalisation: ``backtest_strategy`` cross-sectionally z-scores the factor
signals before calling ``calc``, so ``calc`` must NOT normalise again — it builds
features from the already-normalised signals, exactly as training did.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quant_fund_agent.modeling.features import predict_to_signal
from quant_fund_agent.strategies.base import BaseStrategy, ModelType


def _coerce_model_type(model_type: str) -> ModelType:
    try:
        return ModelType(model_type)
    except ValueError:
        return ModelType.CUSTOM


class ModelStrategy(BaseStrategy):
    """Strategy backed by a fitted return-prediction model."""

    def __init__(
        self,
        strategy_id: str,
        name: str,
        description: str,
        factor_ids: list[str],
        model_type: str,
        estimator: Any,
        target_horizon: int = 6,
        holding_period: int = 6,
        max_positions: int = 20,
        equal_weight: bool = False,
        min_conviction: float = 0.0,
        position_construction: str = "cross_sectional",
        position_params: dict | None = None,
        executor_id: str | None = None,
        artifact_path: str = "",
    ) -> None:
        self.strategy_id = strategy_id
        self.name = name
        self.description = description
        self.factor_ids = list(factor_ids)
        self.model_type = _coerce_model_type(model_type)
        self._model_type_str = model_type
        self.estimator = estimator
        self.target_horizon = target_horizon
        self.holding_period = holding_period
        self.max_positions = max_positions
        self.equal_weight = equal_weight
        self.min_conviction = min_conviction
        self.position_construction = position_construction
        self.position_params = dict(position_params or {})
        self.executor_id = executor_id
        self.model_params = {
            "model_type": model_type,
            "target_horizon": target_horizon,
            "artifact_path": artifact_path,
        }

    @classmethod
    def from_artifact(
        cls,
        artifact_path: str | Path,
        *,
        strategy_id: str,
        name: str = "",
        description: str = "",
        holding_period: int = 6,
        max_positions: int = 20,
        equal_weight: bool = False,
        min_conviction: float = 0.0,
        position_construction: str = "cross_sectional",
        position_params: dict | None = None,
        executor_id: str | None = None,
    ) -> "ModelStrategy":
        """Rebuild a runnable strategy from a persisted joblib artifact.

        The position-construction regime is NOT stored in the artifact (it is a
        deployment choice, not part of the fitted model); callers pass it from the
        persisted ``StrategyRecord`` so OOS / live reproduce the same positions.
        """
        import joblib

        blob = joblib.load(artifact_path)
        return cls(
            strategy_id=strategy_id,
            name=name or strategy_id,
            description=description,
            factor_ids=list(blob["factor_ids"]),
            model_type=blob["model_type"],
            estimator=blob["estimator"],
            target_horizon=int(blob.get("target_horizon", 6)),
            holding_period=holding_period,
            max_positions=max_positions,
            equal_weight=equal_weight,
            min_conviction=min_conviction,
            position_construction=position_construction,
            position_params=position_params,
            executor_id=executor_id,
            artifact_path=str(artifact_path),
        )

    def calc(
        self,
        factor_signals: dict[str, pd.DataFrame],
        data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        close = data["close"]
        return predict_to_signal(
            self.estimator,
            factor_signals,
            self.factor_ids,
            index=close.index,
            columns=close.columns,
        )
