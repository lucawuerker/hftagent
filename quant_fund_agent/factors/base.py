"""Abstract base class for all factor implementations.

Every factor is a Python class that:
  1. Declares metadata as class attributes (``factor_id``, ``name``, …).
  2. Implements a ``calc`` method that turns raw OHLCV data into a signal
     DataFrame of the same shape.

The ``data`` argument passed to ``calc`` follows the convention::

    data = {
        "open":   pd.DataFrame,   # index=dates, columns=tickers
        "high":   pd.DataFrame,
        "low":    pd.DataFrame,
        "close":  pd.DataFrame,
        "volume": pd.DataFrame,
    }

``calc`` must return a ``pd.DataFrame`` with the same index/columns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseFactor(ABC):
    # -- subclasses MUST override these class attributes --
    factor_id: str = ""
    name: str = ""
    category: str = ""  # e.g. "momentum", "mean_reversion"
    description: str = ""

    # data requirements
    window_length: int = 1
    inputs: list[str] = ["close"]

    # prediction horizon -- in *bars* (the feed's bar size sets the wall-clock
    # meaning: a LOBSTER 10s bar makes horizon 6 ≈ 1 minute; a daily feed makes
    # it 6 trading days).  ``prediction_horizon`` is the bar offset at which this
    # factor's edge is expected to peak; it drives the factor's headline IC and
    # feeds the combined-signal horizon downstream.  ``suggested_horizons`` is an
    # optional set of alternative horizons worth measuring (empty → just the
    # default).  Default 6 keeps the historical global behaviour.
    prediction_horizon: int = 6
    suggested_horizons: list[int] = []

    @abstractmethod
    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Compute the factor signal.

        Args:
            data: dict mapping OHLCV field names to DataFrames
                  (index=dates, columns=tickers).

        Returns:
            Signal DataFrame with the same index and columns as the
            input DataFrames.
        """
        ...

    # convenience -----------------------------------------------------------

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} factor_id={self.factor_id!r}>"
