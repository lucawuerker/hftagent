"""Assemble verdict-card metrics from a dumped strategy-pipeline candidate.

Everything here is deterministic and LLM-free: the numbers come from the
Architect's recorded trial history, the Statistician's test details, and a
per-strategy CSCV PBO computed over the return series of the configurations
the Architect's search actually tried.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quant_fund_agent.pipeline import deserialise_series
from quant_fund_agent.research_eval.deflation import pbo_cscv

DSR_TEST_ID = "deflated_sharpe_ratio"
OOS_TEST_ID = "out_of_sample_backtest"

# Map the OOS test's verdict onto the card's plain-English OOS result.
OOS_RESULT_PHRASE = {
    "pass": "holds",
    "warn": "fades partly",
    "fail": "fades",
    "info": "inconclusive",
}


@dataclass
class CardMetrics:
    """The deterministic numbers a verdict card is built from."""

    strategy_name: str = ""
    model_type: str = ""
    factor_ids: list[str] = field(default_factory=list)
    target_horizon: int | None = None

    is_sharpe: float | None = None          # annualised, Architect final trial
    dsr_prob: float | None = None           # DSR test value (P[true SR > 0])
    dsr_verdict: str | None = None          # pass / warn / fail
    n_trials: int | None = None
    expected_max_sharpe: float | None = None  # annualised, from DSR details

    oos_verdict: str | None = None          # pass / warn / fail / info
    oos_sharpe: float | None = None         # annualised, OOS test details
    oos_is_sharpe: float | None = None      # the IS Sharpe the OOS test compared to
    sharpe_decay_pct: float | None = None   # 100·(1 − OOS/IS) when both positive

    pbo: float | None = None                # per-strategy CSCV PBO
    pbo_n_splits: int = 0

    approved: bool | None = None
    reject_stage: str | None = None

    @property
    def card_ready(self) -> bool:
        """A card needs the harness verdicts — DSR + OOS test results."""
        return self.dsr_prob is not None and self.oos_verdict is not None

    @property
    def oos_result(self) -> str:
        return OOS_RESULT_PHRASE.get((self.oos_verdict or "").lower(), "n/a")


def _test_result(candidate: dict[str, Any], test_id: str) -> dict[str, Any] | None:
    stat = candidate.get("stat_result") or {}
    for r in stat.get("test_results", []):
        if r.get("test_id") == test_id:
            return r
    return None


def per_strategy_pbo(
    trial_history: list[dict[str, Any]], n_groups: int = 8,
) -> dict[str, Any]:
    """CSCV PBO over the configurations the Architect's search tried.

    Each trial's in-sample per-bar return series becomes one column of the
    T×N performance matrix (inner-joined on timestamps).  Interpreting the
    result: the probability that picking the backtest winner among the tried
    variants picked one that underperforms out-of-sample.
    """
    cols: list[pd.Series] = []
    for i, t in enumerate(trial_history or []):
        s = deserialise_series((t.get("metrics") or {}).get("portfolio_returns"))
        if s is not None and len(s) > 0:
            cols.append(s.rename(f"trial_{i}"))
    if len(cols) < 2:
        return {"pbo": None, "n_splits": 0, "n_configs": len(cols)}
    M = pd.concat(cols, axis=1, join="inner").dropna(how="all")
    if M.shape[0] < n_groups:
        return {"pbo": None, "n_splits": 0, "n_configs": M.shape[1]}
    out = pbo_cscv(M.to_numpy(), n_groups=n_groups)
    return {"pbo": out.get("pbo"), "n_splits": out.get("n_splits", 0),
            "n_configs": M.shape[1]}


def batch_pbo(candidates: list[dict[str, Any]], n_groups: int = 8) -> dict[str, Any]:
    """CSCV PBO across the batch: one column per attempt's final IS returns."""
    cols: list[pd.Series] = []
    for i, c in enumerate(candidates):
        s = deserialise_series((c.get("backtest_metrics") or {}).get("portfolio_returns"))
        if s is not None and len(s) > 0:
            cols.append(s.rename(f"attempt_{i}"))
    if len(cols) < 2:
        return {"pbo": None, "n_splits": 0, "n_configs": len(cols)}
    M = pd.concat(cols, axis=1, join="inner").dropna(how="all")
    if M.shape[0] < n_groups:
        return {"pbo": None, "n_splits": 0, "n_configs": M.shape[1]}
    out = pbo_cscv(M.to_numpy(), n_groups=n_groups)
    return {"pbo": out.get("pbo"), "n_splits": out.get("n_splits", 0),
            "n_configs": M.shape[1]}


def assemble_card_metrics(candidate: dict[str, Any]) -> CardMetrics:
    """Pull every card number out of one dumped attempt."""
    spec = candidate.get("spec") or {}
    is_metrics = candidate.get("backtest_metrics") or {}

    m = CardMetrics(
        strategy_name=spec.get("strategy_name") or "Unnamed strategy",
        model_type=spec.get("model_type") or "static_weights",
        factor_ids=list(spec.get("factor_ids") or spec.get("weights") or []),
        target_horizon=spec.get("target_horizon"),
        is_sharpe=is_metrics.get("sharpe_ratio"),
        approved=candidate.get("approved"),
        reject_stage=candidate.get("reject_stage"),
    )

    dsr = _test_result(candidate, DSR_TEST_ID)
    if dsr is not None:
        m.dsr_prob = dsr.get("value")
        m.dsr_verdict = dsr.get("verdict")
        details = dsr.get("details") or {}
        m.n_trials = details.get("n_trials")
        m.expected_max_sharpe = details.get("expected_max_sharpe_annualised")

    oos = _test_result(candidate, OOS_TEST_ID)
    if oos is not None:
        m.oos_verdict = oos.get("verdict")
        details = oos.get("details") or {}
        m.oos_sharpe = details.get("oos_sharpe")
        m.oos_is_sharpe = details.get("is_sharpe")
        if m.oos_sharpe is not None and m.oos_is_sharpe and m.oos_is_sharpe > 0:
            m.sharpe_decay_pct = round(100.0 * (1.0 - m.oos_sharpe / m.oos_is_sharpe), 1)

    p = per_strategy_pbo(candidate.get("trial_history") or [])
    m.pbo, m.pbo_n_splits = p["pbo"], p["n_splits"]
    if m.n_trials is None:
        m.n_trials = len(candidate.get("trial_history") or []) or None
    return m


# ---------------------------------------------------------------------------
# Equity curves (IS from the recorded series; OOS recomputed exactly like the
# Statistician's out-of-sample test — frozen artifact, held-out slice)
# ---------------------------------------------------------------------------

def _cumulative(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod() - 1.0


def recompute_equity_curves(
    candidate: dict[str, Any],
    sharpe_tolerance: float = 0.15,
) -> dict[str, Any]:
    """IS + OOS cumulative-return curves for the card chart.

    IS comes from the Architect's recorded per-bar returns.  OOS is recomputed
    by rebuilding the *frozen* strategy on the held-out slice — the exact
    recipe of ``statistics/tests/out_of_sample.py`` — and cross-checked against
    the Sharpe the Statistician recorded (``recompute_match``).  The OOS curve
    is re-based to continue from the IS terminal value so the two segments
    stitch into one line.
    """
    import uuid

    from quant_fund_agent.agents.architect.state import StrategySpec
    from quant_fund_agent.backtesting.data_split import split_panel, split_signals
    from quant_fund_agent.backtesting.strategy_backtester import backtest_strategy
    from quant_fund_agent.factors import discover_factors
    from quant_fund_agent.modeling.service import _factor_signal, _load_panel
    from quant_fund_agent.strategies.dynamic import DynamicStrategy

    is_returns = deserialise_series(
        (candidate.get("backtest_metrics") or {}).get("portfolio_returns"))
    if is_returns is None or is_returns.empty:
        raise ValueError("candidate has no recorded in-sample returns")
    is_curve = _cumulative(is_returns)

    discover_factors()
    spec = StrategySpec.model_validate(candidate["spec"])
    feature_ids = spec.factor_ids or list(spec.weights.keys())
    oos_ratio = float(candidate.get("oos_split_ratio") or 0.2)

    data_full = _load_panel()
    signals_full = {fid: _factor_signal(fid, data_full) for fid in feature_ids}
    _, data_oos = split_panel(data_full, oos_ratio)
    _, signals_oos = split_signals(signals_full, oos_ratio)

    if spec.model_type and spec.model_type != "static_weights":
        from quant_fund_agent.strategies.model_strategy import ModelStrategy

        if not spec.model_artifact_path:
            raise ValueError("ML spec has no persisted model artifact")
        strategy = ModelStrategy.from_artifact(
            spec.model_artifact_path,
            strategy_id=f"card_oos_{uuid.uuid4().hex[:8]}",
            name=spec.strategy_name + " [card OOS]",
            description=candidate.get("hypothesis", ""),
            holding_period=spec.holding_period,
            max_positions=spec.max_positions,
            equal_weight=spec.equal_weight,
            min_conviction=spec.min_conviction,
            position_construction=spec.position_construction,
            position_params=spec.position_params,
        )
    else:
        strategy = DynamicStrategy(
            strategy_id=f"card_oos_{uuid.uuid4().hex[:8]}",
            name=spec.strategy_name + " [card OOS]",
            description=candidate.get("hypothesis", ""),
            factor_ids=list(spec.weights.keys()),
            weights=spec.weights,
            holding_period=spec.holding_period,
            max_positions=spec.max_positions,
            equal_weight=spec.equal_weight,
            min_conviction=spec.min_conviction,
            position_construction=spec.position_construction,
            position_params=spec.position_params,
        )
    oos_result = backtest_strategy(strategy, signals_oos, data_oos)
    oos_returns = oos_result.portfolio_returns

    recorded = _test_result(candidate, OOS_TEST_ID) or {}
    recorded_sharpe = (recorded.get("details") or {}).get("oos_sharpe")
    recomputed_sharpe = float(oos_result.metrics.sharpe_ratio or 0.0)
    match = (recorded_sharpe is None
             or abs(recomputed_sharpe - float(recorded_sharpe))
             <= sharpe_tolerance * max(1.0, abs(float(recorded_sharpe))))

    # Re-base the OOS cumulative curve to continue from the IS terminal value.
    is_end = float(is_curve.iloc[-1])
    oos_curve = (1.0 + is_end) * (1.0 + _cumulative(oos_returns)) - 1.0

    return {
        "is_curve": is_curve,
        "oos_curve": oos_curve,
        "is_returns": is_returns,
        "oos_returns": oos_returns,
        "oos_sharpe_recomputed": recomputed_sharpe,
        "oos_sharpe_recorded": recorded_sharpe,
        "recompute_match": bool(match),
    }


# ---------------------------------------------------------------------------
# SVG polyline scaling (the landing page's 600×260 viewBox, divider at x=340)
# ---------------------------------------------------------------------------

VIEW_W, VIEW_H = 600, 260
X_MIN, X_DIVIDER, X_MAX = 20, 340, 580
Y_TOP, Y_BOTTOM = 40, 230


def _downsample(series: pd.Series, n_points: int) -> pd.Series:
    if len(series) <= n_points:
        return series
    idx = [round(i * (len(series) - 1) / (n_points - 1)) for i in range(n_points)]
    return series.iloc[idx]


def svg_points(
    is_curve: pd.Series,
    oos_curve: pd.Series,
    n_is: int = 9,
    n_oos: int = 7,
) -> dict[str, str]:
    """Scale the stitched curve into the card SVG's coordinate system.

    Returns ``points_is`` / ``points_oos`` strings that drop straight into the
    landing page's ``<polyline points="...">``.  The OOS polyline starts at the
    IS polyline's final point (the divider), like the page's own markup.
    """
    is_pts = _downsample(is_curve, n_is)
    oos_pts = _downsample(oos_curve, n_oos)
    lo = min(float(is_pts.min()), float(oos_pts.min()))
    hi = max(float(is_pts.max()), float(oos_pts.max()))
    span = (hi - lo) or 1.0

    def _y(v: float) -> float:
        return round(Y_BOTTOM - (v - lo) / span * (Y_BOTTOM - Y_TOP), 1)

    def _xs(n: int, x0: float, x1: float) -> list[float]:
        if n == 1:
            return [x1]
        return [round(x0 + i * (x1 - x0) / (n - 1), 1) for i in range(n)]

    is_xy = list(zip(_xs(len(is_pts), X_MIN, X_DIVIDER),
                     [_y(float(v)) for v in is_pts]))
    # OOS continues from the IS end point at the divider.
    oos_xs = _xs(len(oos_pts) + 1, X_DIVIDER, X_MAX)[1:]
    oos_xy = [is_xy[-1]] + list(zip(oos_xs, [_y(float(v)) for v in oos_pts]))

    def _fmt(xy: list[tuple[float, float]]) -> str:
        return " ".join(f"{x:g},{y:g}" for x, y in xy)

    return {"points_is": _fmt(is_xy), "points_oos": _fmt(oos_xy)}
