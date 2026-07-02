"""Compile factor source **in-memory** — no file, no registry entry.

The evolutionary factor researcher evaluates hundreds of candidate programs per
run.  Materialising each one the oneshot way (write into the shared
``factors/researcher/`` package, import, register) would churn files and pollute
the global registry with transient candidates — and the registry raises on
duplicate ids, so a jitter probe reusing its parent's id could never register at
all.  This module gives the evolution loop a side-effect-free alternative:

* :func:`compile_factor` — run the full static validator
  (:func:`~quant_fund_agent.agents.factor_research.codegen.validate_code`), then
  ``exec`` the module source in an isolated namespace and hand back the factor
  *class*.  The ``@register_factor`` decorator inside the source still fires (the
  code is byte-identical to what would be persisted), but the global registry is
  restored to its prior state before returning, so nothing leaks.
* :func:`compute_signal` — instantiate a compiled class and run ``calc`` on a
  panel restricted to the factor's declared ``inputs``.

Only candidates that survive the whole evolutionary run are persisted, via the
same :func:`~quant_fund_agent.agents.factor_research.codegen.materialise` path
the oneshot engine uses — so a persisted factor is always a real, importable
file and the two engines cannot drift.
"""

from __future__ import annotations

import logging

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor

log = logging.getLogger("factors.inmem")


def compile_factor(code: str, factor_id: str, *,
                   smoke: bool = False) -> type[BaseFactor]:
    """Validate + exec factor source and return the class, leaving no trace.

    Runs the exact static checks the persist path runs (imports allow-list,
    shape, ``inputs`` coverage, ``prediction_horizon``), so code that compiles
    here will also materialise later.  ``smoke=True`` additionally runs ``calc``
    on the same synthetic panel the oneshot materialise path uses — catching
    runtime slips (a missing import, a NameError) at *codegen* time, where the
    retry-with-feedback prompt can fix them, instead of at evaluation time.
    Raises
    :class:`~quant_fund_agent.agents.factor_research.codegen.CodeValidationError`
    on any validation/smoke failure and whatever the source raises on exec.
    """
    from quant_fund_agent.agents.factor_research.codegen import (
        CodeValidationError,
        validate_code,
    )
    from quant_fund_agent.factors.registry import _FACTOR_REGISTRY

    class_name = validate_code(code, factor_id)

    prior = _FACTOR_REGISTRY.pop(factor_id, None)  # let re-registration through
    namespace: dict = {"__name__": f"<inmem:{factor_id}>"}
    try:
        exec(compile(code, f"<factor:{factor_id}>", "exec"), namespace)  # noqa: S102
        cls = namespace[class_name]
    finally:
        # Restore the registry exactly as it was: in-memory candidates are
        # transient and must never shadow or displace a persisted factor.
        _FACTOR_REGISTRY.pop(factor_id, None)
        if prior is not None:
            _FACTOR_REGISTRY[factor_id] = prior

    if not issubclass(cls, BaseFactor):
        raise CodeValidationError(f"{class_name} is not a BaseFactor subclass")

    if smoke:
        from quant_fund_agent.agents.factor_research.codegen import (
            _make_synthetic_panel,
        )

        try:
            compute_signal(cls, _make_synthetic_panel())
        except Exception as e:  # noqa: BLE001 — feed straight into the retry prompt
            raise CodeValidationError(
                f"calc() raised on synthetic data: {e}") from e
    return cls


def compute_signal(cls: type[BaseFactor], panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run a compiled factor's ``calc`` on ``panel`` → its signal frame.

    The panel is passed through as-is (factors read only their declared
    ``inputs``; a missing field raises ``KeyError`` exactly like the oneshot
    backtest, which the caller reports as an evaluation failure).  The output is
    validated to be a DataFrame aligned with ``panel['close']``.
    """
    instance = cls()
    out = instance.calc(panel)
    if not isinstance(out, pd.DataFrame):
        raise TypeError(f"calc() must return a DataFrame, got {type(out).__name__}")
    close = panel["close"]
    if out.shape != close.shape:
        # Reindex rather than fail: several seed factors return a sub-grid after
        # defensive dropna; alignment onto the close grid is the shared convention.
        out = out.reindex(index=close.index, columns=close.columns)
    return out


def signal_from_code(code: str, factor_id: str,
                     panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Convenience: :func:`compile_factor` + :func:`compute_signal` in one call."""
    return compute_signal(compile_factor(code, factor_id), panel)
