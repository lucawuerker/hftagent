"""The ``BaseExecutor`` contract — one evolution unit of the execution layer.

An **executor** is the program that turns a strategy's composite signal into a
target book through time (the signal → positions mapping that used to be
hardcoded in ``backtesting/strategy_backtester._signal_to_positions`` and
``backtesting/positions.per_underlying_positions``).  It mirrors ``BaseFactor``
exactly: a registered class with declared metadata, validated at codegen,
compiled in-memory during evolution, persisted only for survivors.

Contract (see ``docs/execution-evolution/DESIGN.md`` §Genome):

* ``step(t, signal_row, state_row, book) -> pd.Series`` — the canonical,
  always-available stepwise API: one bar's signal + causal state + the
  executor's **own current book** → target weights row.  Path-dependent logic
  (stops, profit-taking, position-aware rebalancing) is first-class.
* ``target_weights(signal, state) -> pd.DataFrame`` — OPTIONAL vectorised
  fast-path for path-INdependent programs; the harness uses it when
  implemented, else drives ``step`` bar-by-bar via :func:`run_executor`.

Weight conventions (validated by :func:`validate_weights`, not trusted):

* ``NaN`` weights are **treated as 0** by the harness (the seed programs
  legitimately emit NaN before z-score warm-up and on all-zero rows) — but
  ``±inf``, per-name bound breaches, gross-leverage breaches and (for the
  ``cross_sectional`` regime) dollar-neutrality violations fail validity.
* Weights are portfolio fractions: ``|w[t, i]| ≤ max_name`` and
  ``Σ_i |w[t, i]| ≤ max_gross`` per bar.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("execution.base")

VALID_REGIMES = ("cross_sectional", "per_underlying")

# Default output-contract bounds (harness-side; identical for every candidate).
DEFAULT_MAX_GROSS = 2.0        # Σ|w| per bar
DEFAULT_MAX_NAME = 1.0         # |w| per name
# |Σw| tolerance for cross_sectional books.  Deliberately loose: the baseline
# top-K construction is only *approximately* neutral (selecting the K largest
# |z| breaks exact zero-sum), and genome #0 must satisfy its own contract.
# Tighten via ExecEvalParams.neutral_tol for stricter runs.
DEFAULT_NEUTRAL_TOL = 0.50


class BaseExecutor:
    """Base class every execution program subclasses (the genome unit).

    Class attributes (validated by ``execution.codegen.validate_executor_code``):

    * ``executor_id`` — unique snake_case id.
    * ``name`` / ``description`` — human-readable metadata.
    * ``regime`` — which book shape it builds: ``"cross_sectional"``
      (dollar-neutral long/short) or ``"per_underlying"`` (directional).
    * ``inputs`` — the state/panel fields it reads (``"signal"`` is implicit);
      used for capability gating exactly like ``BaseFactor.inputs``.
    * ``params`` — dict of numeric constants: the **jitter surface** for the
      window-jitter mutation operator and the plateau probe.
    """

    executor_id: str = ""
    name: str = ""
    description: str = ""
    regime: str = "per_underlying"
    inputs: list[str] = ["signal"]  # noqa: RUF012 — mirrors BaseFactor's style
    params: dict[str, float] = {}   # noqa: RUF012

    # ── canonical stepwise API ────────────────────────────────────────────────
    def step(self, t: int, signal_row: pd.Series, state_row: dict[str, pd.Series],
             book: "BookState") -> pd.Series:
        """One bar: signal + causal state + CURRENT BOOK → target weights row.

        ``signal_row`` is the composite signal at bar ``t`` (index = tickers);
        ``state_row`` maps each state field (``vol``, ``adv``, ``drawdown``,
        ``signal_age``, …) to its bar-``t`` row; ``book`` carries the executor's
        own current positions and per-name unrealised P&L.  Must be causal —
        the harness *proves* this with a truncation-replay probe.
        """
        raise NotImplementedError(
            f"{type(self).__name__} implements neither step() nor target_weights()."
        )

    # ── optional vectorised fast-path ─────────────────────────────────────────
    def target_weights(self, signal: pd.DataFrame,
                       state: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """(T × N) signal + state → (T × N) weights in one shot (path-independent).

        Raise ``NotImplementedError`` (the default) to fall back to the
        bar-by-bar ``step`` loop.
        """
        raise NotImplementedError

    def has_vectorised_path(self) -> bool:
        """True when the class overrides :meth:`target_weights`."""
        return type(self).target_weights is not BaseExecutor.target_weights


# ── registry (mirrors factors.registry) ───────────────────────────────────────

EXECUTOR_REGISTRY: dict[str, type[BaseExecutor]] = {}


def register_executor(cls: type[BaseExecutor]) -> type[BaseExecutor]:
    """Class decorator adding an executor to the global registry.

    Unlike the factor registry this *allows* re-registration of the same id
    with the identical class object (idempotent imports), but raises when a
    different class tries to claim an existing id.
    """
    eid = getattr(cls, "executor_id", "") or ""
    if not eid:
        raise ValueError(f"{cls.__name__} must declare a non-empty executor_id")
    prior = EXECUTOR_REGISTRY.get(eid)
    if prior is not None and prior is not cls:
        raise ValueError(f"executor_id '{eid}' is already registered by "
                         f"{prior.__name__}")
    EXECUTOR_REGISTRY[eid] = cls
    return cls


def get_executor(executor_id: str) -> type[BaseExecutor]:
    """Look up a registered executor class (imports the seeds on first use)."""
    if executor_id not in EXECUTOR_REGISTRY:
        # Seeds register on import; make the common path just work.
        from quant_fund_agent.execution import seeds  # noqa: F401
    try:
        return EXECUTOR_REGISTRY[executor_id]
    except KeyError:
        raise KeyError(
            f"unknown executor_id '{executor_id}' "
            f"(registered: {sorted(EXECUTOR_REGISTRY)})") from None


def list_executors() -> list[str]:
    from quant_fund_agent.execution import seeds  # noqa: F401 — ensure seeds present
    return sorted(EXECUTOR_REGISTRY)


# ── output-contract validation (validated, never trusted) ─────────────────────

def validate_weights(
    weights: pd.DataFrame,
    regime: str,
    *,
    max_gross: float = DEFAULT_MAX_GROSS,
    max_name: float = DEFAULT_MAX_NAME,
    neutral_tol: float = DEFAULT_NEUTRAL_TOL,
) -> list[str]:
    """Check a weight frame against the hard output contract → list of violations.

    Empty list = valid.  ``NaN`` cells are the documented "no position"
    convention (treated as 0 downstream) and are NOT violations; ``±inf`` is.
    Dollar-neutrality is only enforced for the ``cross_sectional`` regime, and
    only on bars that hold any position at all.
    """
    problems: list[str] = []
    w = weights.to_numpy(dtype=float)
    if np.isinf(w).any():
        problems.append("non-finite (inf) weights")
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

    max_abs = float(np.max(np.abs(w))) if w.size else 0.0
    if max_abs > max_name + 1e-9:
        problems.append(f"per-name bound breached: max |w|={max_abs:.4f} > {max_name}")

    gross = np.abs(w).sum(axis=1)
    if gross.size and float(gross.max()) > max_gross + 1e-9:
        problems.append(
            f"gross-leverage bound breached: max Σ|w|={float(gross.max()):.4f} > {max_gross}")

    if regime == "cross_sectional":
        net = w.sum(axis=1)
        active = gross > 1e-12
        if active.any():
            worst = float(np.max(np.abs(net[active])))
            if worst > neutral_tol + 1e-9:
                problems.append(
                    f"dollar-neutrality violated: max |Σw|={worst:.4f} > tol={neutral_tol}")
    return problems


# ── book state (the path-dependence carrier for the stepwise loop) ────────────

class BookState:
    """The executor's own live book during a stepwise run.

    Tracks current positions (weights), a per-name average entry price, the
    per-name **unrealised P&L** (in weight·return units) and the strategy's own
    running equity/drawdown — everything locked decision 7 of the DESIGN says an
    executor may condition on.  Updated by :func:`run_executor` *before* each
    ``step`` call with the bar's prices, so ``step`` always sees a causal,
    up-to-date view.
    """

    def __init__(self, columns: pd.Index):
        self.columns = columns
        self.positions = pd.Series(0.0, index=columns)
        self.entry_price = pd.Series(np.nan, index=columns)
        self.unrealised_pnl = pd.Series(0.0, index=columns)
        self.equity = 0.0            # cumulative (sum of per-bar book returns)
        self.peak_equity = 0.0
        self.drawdown = 0.0          # ≤ 0
        self._last_price: pd.Series | None = None

    def mark(self, price_row: pd.Series) -> None:
        """Mark the book to the new bar's prices (called once per bar, pre-step)."""
        price = price_row.reindex(self.columns)
        if self._last_price is not None:
            ret = (price / self._last_price - 1.0).replace([np.inf, -np.inf], np.nan)
            pnl = (self.positions * ret).fillna(0.0)
            self.equity += float(pnl.sum())
            self.peak_equity = max(self.peak_equity, self.equity)
            self.drawdown = self.equity - self.peak_equity
            held = self.positions.abs() > 1e-12
            upl = (self.positions * (price / self.entry_price - 1.0)).where(held)
            self.unrealised_pnl = upl.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self._last_price = price

    def rebalance(self, new_weights: pd.Series) -> None:
        """Adopt the step's target weights (entry price resets on sign flip / new)."""
        new = new_weights.reindex(self.columns).fillna(0.0)
        if self._last_price is not None:
            opened = (new.abs() > 1e-12) & (
                (self.positions.abs() <= 1e-12) | (np.sign(new) != np.sign(self.positions))
            )
            self.entry_price = self.entry_price.where(~opened, self._last_price)
        self.entry_price = self.entry_price.where(new.abs() > 1e-12, np.nan)
        self.positions = new


# ── the driver ─────────────────────────────────────────────────────────────────

def run_executor(
    executor: BaseExecutor,
    signal: pd.DataFrame,
    state: dict[str, pd.DataFrame],
    close: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run an executor over a signal panel → (T × N) target-weight frame.

    Uses the vectorised :meth:`BaseExecutor.target_weights` fast-path when the
    class implements it; otherwise drives :meth:`BaseExecutor.step` bar-by-bar,
    maintaining a :class:`BookState` marked to ``close`` (required for the
    stepwise path).  Output is reindexed onto the signal grid; the caller
    validates the contract with :func:`validate_weights`.
    """
    if executor.has_vectorised_path():
        out = executor.target_weights(signal, state)
        if not isinstance(out, pd.DataFrame):
            raise TypeError(
                f"target_weights() must return a DataFrame, got {type(out).__name__}")
        return out.reindex(index=signal.index, columns=signal.columns)

    if close is None:
        raise ValueError("stepwise execution needs `close` to mark the book")
    book = BookState(signal.columns)
    rows: list[pd.Series] = []
    state_items = list(state.items())
    for t, ts in enumerate(signal.index):
        book.mark(close.loc[ts])
        state_row = {k: df.loc[ts] for k, df in state_items}
        w = executor.step(t, signal.loc[ts], state_row, book)
        if not isinstance(w, pd.Series):
            raise TypeError(f"step() must return a Series, got {type(w).__name__}")
        w = w.reindex(signal.columns)
        book.rebalance(w)
        rows.append(w)
    return pd.DataFrame(rows, index=signal.index, columns=signal.columns)


def resolve_executor(override: str | None = None) -> str | None:
    """Resolve the deployment executor: override > ``QF_EXECUTOR`` > None.

    ``None`` / ``"auto"`` / ``"legacy"`` → the legacy position-construction
    regimes (byte-identical to pre-E4 behaviour).  A named executor must be
    registered (evolved executors register once their code is materialised).
    """
    val = (override or os.getenv("QF_EXECUTOR") or "").strip()
    if not val or val.lower() in ("none", "auto", "legacy"):
        return None
    get_executor(val)   # raises KeyError for an unknown id — fail fast
    return val


def executor_spec(cls: type[BaseExecutor]) -> dict[str, Any]:
    """JSON-safe summary of an executor class (for manifests / SOTA state)."""
    return {
        "executor_id": cls.executor_id,
        "name": cls.name or cls.__name__,
        "regime": cls.regime,
        "inputs": list(cls.inputs or []),
        "params": dict(cls.params or {}),
        "vectorised": cls.target_weights is not BaseExecutor.target_weights,
    }
