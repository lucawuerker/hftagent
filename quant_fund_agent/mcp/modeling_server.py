"""MCP server exposing the Architect's pre-implemented model toolbox.

Run as a stdio server::

    python -m quant_fund_agent.mcp.modeling_server

It owns the (heavy) data panel server-side and caches per-factor signals, so the
panel is loaded once and reused across every architect iteration.  Only small
JSON crosses the MCP boundary: model specs, the chosen hyper-parameters, and the
resulting metrics + artifact path.

All real work is delegated to :mod:`quant_fund_agent.modeling.service`, so the
in-process fallback in ``client.py`` computes byte-for-byte identical results.

IMPORTANT: stdout is the MCP protocol channel — every tool returns a JSON string
and all logging goes to stderr.  Never ``print`` to stdout here.
"""

from __future__ import annotations

import json
import logging
import sys

from mcp.server.fastmcp import FastMCP

from quant_fund_agent.modeling.service import model_menu, run_fit_and_backtest

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s  %(levelname)-7s  %(name)-20s  %(message)s")
log = logging.getLogger("mcp.modeling_server")

mcp = FastMCP("quant-modeling")


@mcp.tool()
def list_models() -> str:
    """List the fittable models and their tunable hyper-parameters (JSON).

    Returns a JSON array; each entry has ``model_type``, ``family``,
    ``description`` and a ``params`` list with each hyper-parameter's type,
    default and valid range.  Includes the ``static_weights`` baseline.
    """
    return json.dumps(model_menu())


@mcp.tool()
def fit_and_backtest(
    model_type: str,
    factor_ids: list[str] | None = None,
    model_params: dict | None = None,
    weights: dict | None = None,
    target_horizon: int = 6,
    holding_period: int = 6,
    max_positions: int = 20,
    equal_weight: bool = False,
    min_conviction: float = 0.0,
    position_construction: str = "cross_sectional",
    position_params: dict | None = None,
    executor_id: str | None = None,
    oos_split_ratio: float = 0.2,
    valid_ratio: float = 0.3,
    strategy_id: str = "",
    as_of: str | None = None,
) -> str:
    """Fit ``model_type`` on the in-sample window and backtest it (JSON result).

    For ML models, pass ``factor_ids`` (features) and ``model_params``; the model
    is fit on an IS-train sub-window and back-tested on the held-out IS-valid
    sub-window.  For ``static_weights`` pass ``weights`` instead.  The held-out
    OOS slice (``oos_split_ratio``) is never shown to the model.

    ``as_of`` (walk-forward backtest): ISO date string; when set, only data
    strictly before it is visible to the fit/backtest.

    Returns JSON: ``{model_type, factor_ids, metrics, artifact_path, diagnostics}``.
    """
    log.info("fit_and_backtest model_type=%s factors=%s", model_type,
             factor_ids if model_type != "static_weights" else list((weights or {}).keys()))
    result = run_fit_and_backtest(
        model_type=model_type,
        factor_ids=factor_ids,
        model_params=model_params,
        weights=weights,
        target_horizon=target_horizon,
        holding_period=holding_period,
        max_positions=max_positions,
        equal_weight=equal_weight,
        min_conviction=min_conviction,
        position_construction=position_construction,
        position_params=position_params,
        executor_id=executor_id,
        oos_split_ratio=oos_split_ratio,
        valid_ratio=valid_ratio,
        strategy_id=strategy_id or None,
        as_of=as_of,
    )
    return json.dumps(result)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
