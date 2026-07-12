"""Abstract base class for all strategy implementations.

A strategy combines one or more alpha factors into a composite trading
signal.  The combination method is intentionally unconstrained — it can
be a static linear blend, an OLS/ridge regression, an XGBoost model,
a neural network, etc.

Contract
--------
``calc`` receives pre-computed factor signals and raw OHLCV data and
must return a **signal DataFrame** (index=timestamps, columns=tickers)
where:

    - positive values → long conviction
    - negative values → short conviction
    - magnitude       → relative conviction strength

The backtester will normalise these into dollar-neutral positions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import pandas as pd


class ModelType(str, Enum):
    """Describes how the strategy combines factors."""
    STATIC_WEIGHTS = "static_weights"
    LINEAR_REGRESSION = "linear_regression"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    NEURAL_NETWORK = "neural_network"
    CUSTOM = "custom"


class BaseStrategy(ABC):
    # -- subclasses MUST override these --
    strategy_id: str = ""
    name: str = ""
    description: str = ""

    factor_ids: list[str] = []
    model_type: ModelType = ModelType.STATIC_WEIGHTS

    # how many bars the strategy holds a position before re-scoring
    holding_period: int = 1

    # position construction parameters
    max_positions: int = 20
    equal_weight: bool = False
    min_conviction: float = 0.0

    # How the composite signal becomes positions (see backtesting/positions.py):
    #   "cross_sectional" — rank the cross-section → dollar-neutral long/short
    #                       (the default; uses max_positions/equal_weight/min_conviction);
    #   "per_underlying"  — directional per-name boundary (long/flat/short), each
    #                       active position sized 1/max_positions (uses position_params:
    #                       mode/threshold/zscore_basis/zscore_window).
    position_construction: str = "cross_sectional"
    position_params: dict[str, Any] = {}

    # E4: when set, the book is built by this REGISTERED executor program
    # (execution/base.py registry) instead of the two hardcoded regimes above —
    # stamped once on the StrategySpec/StrategyRecord so Architect IS-fit,
    # Statistician OOS and live reproduce the identical book.  None = legacy.
    executor_id: str | None = None

    # free-form dict for weights, sklearn model, hyper-params, etc.
    model_params: dict[str, Any] = {}

    @abstractmethod
    def calc(
        self,
        factor_signals: dict[str, pd.DataFrame],
        data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Combine factor signals into a single signal DataFrame.

        Args:
            factor_signals: ``{factor_id: signal_df}`` — each DataFrame
                has index=timestamps, columns=tickers.
            data: raw OHLCV panel dict (same format factors receive).

        Returns:
            Signal DataFrame (index=timestamps, columns=tickers).
            Positive → long, negative → short.
        """
        ...

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} strategy_id={self.strategy_id!r} "
            f"model={self.model_type.value}>"
        )
