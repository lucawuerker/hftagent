"""Statistical-test logic shared by the quant-statistics MCP server and its
in-process fallback.

The Statistician's LLM steps (risk assessment, accept/reject) stay in the agent
graph.  The deterministic work — rebuilding the full ``StatTestContext`` (loading
the panel, computing factor signals, deserialising the IS returns) and running
the registered tests (deflated Sharpe, out-of-sample backtest) — lives here.

The panel and per-factor signals are loaded via :mod:`quant_fund_agent.modeling.service`
so a long-lived server reuses the same cached panel the modeling service uses.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

log = logging.getLogger("statistics.service")


def _deserialise_series(payload: dict | None) -> pd.Series | None:
    if not payload or "timestamps" not in payload:
        return None
    idx = pd.to_datetime(payload["timestamps"])
    return pd.Series(payload["values"], index=idx)


def list_tests() -> dict[str, Any]:
    """Return the registered statistical tests grouped by mandatory/optional."""
    from quant_fund_agent.statistics import (
        discover_tests,
        get_all_test_classes,
        get_mandatory_test_ids,
        get_optional_test_ids,
    )

    discover_tests()
    classes = get_all_test_classes()
    return {
        "mandatory": get_mandatory_test_ids(),
        "optional": [
            {"id": tid, "description": classes[tid].description}
            for tid in get_optional_test_ids()
        ],
    }


def _build_context(
    spec: dict[str, Any],
    is_metrics: dict[str, Any],
    trial_history: list[dict[str, Any]],
    hypothesis: str,
    oos_split_ratio: float,
    bars_per_day: int,
    as_of: str | None = None,
):
    from quant_fund_agent.agents.architect.state import StrategySpec, TrialRecord
    from quant_fund_agent.factors import discover_factors
    from quant_fund_agent.modeling.service import (
        _factor_signal,
        _load_panel,
        _truncate_as_of,
    )
    from quant_fund_agent.statistics import StatTestContext

    discover_factors()
    spec_obj = StrategySpec.model_validate(spec)
    feature_ids = spec_obj.factor_ids or list(spec_obj.weights.keys())

    data_full = _load_panel()
    factor_signals_full = {fid: _factor_signal(fid, data_full) for fid in feature_ids}

    # Walk-forward cutoff: the OOS test must run on the tail of the pre-cutoff
    # window (genuinely held out from the architect AND strictly before the
    # simulator's live trading window), never on post-cutoff data.
    data_full = _truncate_as_of(data_full, as_of)
    factor_signals_full = _truncate_as_of(factor_signals_full, as_of)

    return StatTestContext(
        strategy_spec=spec_obj,
        hypothesis=hypothesis,
        is_portfolio_returns=_deserialise_series(is_metrics.get("portfolio_returns")),
        is_equity_curve=_deserialise_series(is_metrics.get("equity_curve")),
        is_metrics=is_metrics,
        data_full=data_full,
        factor_signals_full=factor_signals_full,
        oos_split_ratio=oos_split_ratio,
        trial_history=[TrialRecord.model_validate(t) for t in trial_history],
        bars_per_day=bars_per_day,
    )


def run_tests(
    test_ids: list[str],
    spec: dict[str, Any],
    is_metrics: dict[str, Any] | None = None,
    trial_history: list[dict[str, Any]] | None = None,
    hypothesis: str = "",
    oos_split_ratio: float = 0.2,
    bars_per_day: int = 2340,
    as_of: str | None = None,
) -> list[dict[str, Any]]:
    """Build the test context server-side and run each requested test.

    Returns a list of ``StatTestResult`` model dumps (one per ``test_id``).  A
    test that raises is reported as a ``FAIL`` result rather than aborting.

    ``as_of`` (walk-forward backtest): ISO date string; when set the OOS test
    only sees data strictly before it.
    """
    from quant_fund_agent.statistics import (
        StatTestResult,
        StatTestVerdict,
        discover_tests,
        instantiate_test,
    )

    discover_tests()
    ctx = _build_context(
        spec=spec,
        is_metrics=is_metrics or {},
        trial_history=trial_history or [],
        hypothesis=hypothesis,
        oos_split_ratio=oos_split_ratio,
        bars_per_day=bars_per_day,
        as_of=as_of,
    )

    results: list[dict[str, Any]] = []
    for test_id in test_ids:
        test = instantiate_test(test_id)
        log.info("  … running %s", test.name)
        try:
            result = test.run(ctx)
        except Exception as e:  # noqa: BLE001 — never crash the agent
            log.exception("Test %s raised: %s", test_id, e)
            result = StatTestResult(
                test_id=test_id,
                test_name=test.name,
                verdict=StatTestVerdict.FAIL,
                summary=f"Test raised an exception: {e}",
            )
        results.append(result.model_dump(mode="json"))
    return results
