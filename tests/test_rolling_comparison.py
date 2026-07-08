"""Offline tests for the rolling-window comparison driver + aggregator.

Everything here is synthetic and fast (no LOBSTER panel, no LLM, no subprocess):
- window generation from ``bin*.csv`` month discovery,
- the per-underlying *time-series* IC (the de-cross-sectionalised track) on a
  SINGLE ticker — the regression that the cross-sectional IC/analytics went
  all-NaN for one name,
- the analytics feature matrix / diversity / importance being finite for one
  ticker,
- the aggregation of fabricated per-run CSVs into combined tables + per-ticker
  feature-importance-over-OOS-months matrices and figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.comparison import ic as cic
from quant_fund_agent.comparison import rolling
from quant_fund_agent.comparison.config import ComparisonConfig
from quant_fund_agent.comparison.standardize import per_underlying_zscore


# ── window generation ─────────────────────────────────────────────────────────

def _touch_months(root: Path, ticker: str, months: list[str]) -> Path:
    d = root / ticker
    d.mkdir(parents=True, exist_ok=True)
    for m in months:
        (d / f"bin{m.replace('-', '')}.csv").write_text("x")
    return d


def test_discover_months_and_rolling_windows(tmp_path):
    months = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"]
    d = _touch_months(tmp_path, "CORN", months)
    assert rolling.discover_months(d) == months

    wins = rolling.build_windows(months, is_len=2, oos_len=1, step=1)
    # 5 months → windows for OOS 03, 04, 05.
    assert [w.label for w in wins] == ["2024-03", "2024-04", "2024-05"]
    assert wins[0].is_spec == "2024-01,2024-02" and wins[0].oos_spec == "2024-03"
    # rolling: the previous OOS month becomes the second IS month.
    assert wins[1].is_months == ("2024-02", "2024-03") and wins[1].oos_months == ("2024-04",)


def test_build_windows_too_few_months():
    assert rolling.build_windows(["2024-01", "2024-02"], is_len=2, oos_len=1) == []


def test_completed_status_gate(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    assert not rolling._completed(d)
    (d / "status.json").write_text(json.dumps({"tracks": {"ic": {"status": "failed"}}}))
    assert not rolling._completed(d)
    (d / "status.json").write_text(json.dumps(
        {"tracks": {"ic": {"status": "ok"}, "bruteforce": {"status": "ok"}}}))
    assert rolling._completed(d)


# ── per-underlying time-series IC on a single ticker (the key regression) ──────

def test_per_underlying_ic_single_ticker_is_finite_and_matches_pearson():
    idx = pd.date_range("2024-01-01", periods=300, freq="min")
    rng = np.random.default_rng(0)
    close = pd.DataFrame({"AAA": 100 + np.cumsum(rng.normal(0, 1, 300))}, index=idx)
    fwd1 = close.shift(-1) / close - 1.0
    # A factor genuinely linearly related to the next-bar return, plus noise.
    sig = pd.DataFrame({"AAA": fwd1["AAA"].fillna(0.0) + rng.normal(0, 5e-4, 300)}, index=idx)
    panel = {"close": close}

    row = cic._per_underlying_ic_row("p", "f", sig, panel, (1,), {})
    assert row["ic_1"] is not None and np.isfinite(row["ic_1"])
    assert row["n_timestamps"] > 100

    # Independent check: Pearson of the standardised factor vs forward return.
    x = per_underlying_zscore(sig)["AAA"]
    y = fwd1["AAA"]
    m = x.notna() & y.notna()
    expected = x[m].corr(y[m], method="pearson")
    assert abs(row["ic_1"] - expected) < 1e-9
    assert row["ic_1"] > 0.5  # the planted relationship is strong


def test_evaluate_prerun_ic_uses_per_underlying_when_cfg_per_underlying(monkeypatch):
    """A single-ticker prerun yields finite IC under the default per-underlying mode."""
    from quant_fund_agent.modeling import service

    idx = pd.date_range("2024-01-01", periods=200, freq="min")
    rng = np.random.default_rng(1)
    close = pd.DataFrame({"AAA": 100 + np.cumsum(rng.normal(0, 1, 200))}, index=idx)
    fwd = close.shift(-1) / close - 1.0
    signals = {
        "f1": pd.DataFrame({"AAA": fwd["AAA"].fillna(0.0)}, index=idx),
        "f2": pd.DataFrame({"AAA": rng.normal(0, 1, 200)}, index=idx),
    }
    panel = {"close": close}
    monkeypatch.setattr(service, "_factor_signal", lambda fid, p: signals[fid])

    cfg = ComparisonConfig(fit_standardize="per_underlying", target_horizon=1)
    rows = cic.evaluate_prerun_ic("p", ["f1", "f2"], panel, (1,), {}, cfg=cfg)
    assert len(rows) == 2
    assert all(r["ic_1"] is not None and np.isfinite(r["ic_1"]) for r in rows)


# ── analytics feature matrix / diversity / importance for one ticker ───────────

def test_analytics_finite_for_single_ticker(monkeypatch):
    from quant_fund_agent.comparison import analytics
    from quant_fund_agent.modeling import service

    idx = pd.date_range("2024-01-01", periods=300, freq="min")
    rng = np.random.default_rng(2)
    close = pd.DataFrame({"AAA": 100 + np.cumsum(rng.normal(0, 1, 300))}, index=idx)
    fwd = close.shift(-1) / close - 1.0
    signals = {
        "f1": pd.DataFrame({"AAA": fwd["AAA"].fillna(0.0) + rng.normal(0, 1e-3, 300)}, index=idx),
        "f2": pd.DataFrame({"AAA": rng.normal(0, 1, 300)}, index=idx),
        "f3": pd.DataFrame({"AAA": np.cumsum(rng.normal(0, 1, 300))}, index=idx),
    }
    panel = {"close": close}
    monkeypatch.setattr(service, "_factor_signal", lambda fid, p: signals[fid])

    cfg = ComparisonConfig(fit_standardize="per_underlying", target_horizon=1,
                           analytics_max_rows=10_000)
    fids = ["f1", "f2", "f3"]

    # Feature matrix must NOT be all-NaN (the cross-sectional path would be, for 1 name).
    X, y = analytics._feature_matrix(fids, panel, cfg, with_target=True)
    assert np.isfinite(X.to_numpy(dtype=float)).any()
    assert y.notna().any()

    summary, factor_rows, corr = analytics.evaluate_prerun_diversity("p", fids, panel, cfg)
    assert summary["eff_n_factors"] is not None
    assert not corr.empty and corr.notna().to_numpy().any()

    mv, imp_rows = analytics.evaluate_prerun_modelview("p", fids, panel, cfg, top_n=10)
    assert imp_rows, "expected non-empty importance rows for a single ticker"
    assert all(np.isfinite(r["importance"]) for r in imp_rows)


# ── aggregation of fabricated per-run outputs ─────────────────────────────────

def _fake_run(batch: Path, ticker: str, oos: str, is_window: str) -> None:
    d = batch / "runs" / ticker / oos
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"is_window": is_window, "oos_window": oos}))
    (d / "status.json").write_text(json.dumps(
        {"tracks": {"ic": {"status": "ok"}, "analytics": {"status": "ok"},
                    "bruteforce": {"status": "ok"}}}))
    preruns = ["gpt4omini120650", "main"]
    models = ["ridge", "ensemble"]
    seed = sum(map(ord, ticker + oos))
    rng = np.random.default_rng(seed)

    pd.DataFrame([
        {"prerun": p, "model": m, "n_factors_used": 5,
         "oos_sharpe": float(rng.normal()), "oos_ic": float(rng.normal(0, 0.05)),
         "is_sharpe": float(rng.normal()), "oos_ann_return": float(rng.normal(0, 0.1)),
         "oos_max_drawdown": float(-abs(rng.normal(0, 0.1)))}
        for p in preruns for m in models
    ]).to_csv(d / "bruteforce_results.csv", index=False)

    pd.DataFrame([
        {"prerun": p, "model": "gradient_boosting", "factor_id": f"{p}_{j}",
         "name": f"factor_{j}", "importance": float(abs(rng.normal())), "rank": j + 1}
        for p in preruns for j in range(4)
    ]).to_csv(d / "analytics_importance.csv", index=False)

    pd.DataFrame([
        {"prerun": p, "factor_id": f"{p}_{j}", "name": f"factor_{j}",
         "ic_1": float(rng.normal(0, 0.05)), "ic_6": float(rng.normal(0, 0.05)),
         "n_timestamps": 1000}
        for p in preruns for j in range(4)
    ]).to_csv(d / "ic_results.csv", index=False)

    pd.DataFrame([
        {"prerun": p, "n_factors": 4, "eff_n_factors": 3.0, "eff_ratio": 0.75,
         "mean_abs_corr": 0.2, "n_clusters": 3, "redundancy": 0.25}
        for p in preruns
    ]).to_csv(d / "analytics_diversity_summary.csv", index=False)


def test_aggregate_builds_combined_and_per_ticker(tmp_path):
    batch = tmp_path / "rolling_test"
    _fake_run(batch, "CORN", "2024-03", "2024-01,2024-02")
    _fake_run(batch, "CORN", "2024-04", "2024-02,2024-03")
    _fake_run(batch, "SPY", "2024-03", "2024-01,2024-02")

    paths = rolling.aggregate(batch)

    bf = pd.read_csv(batch / "combined" / "bruteforce_all.csv")
    assert {"ticker", "oos_month", "is_window"} <= set(bf.columns)
    assert set(bf["ticker"]) == {"CORN", "SPY"}
    assert set(bf["oos_month"]) == {"2024-03", "2024-04"}

    imp = pd.read_csv(batch / "combined" / "importance_all.csv")
    assert {"ticker", "oos_month", "prerun", "model", "name", "importance"} <= set(imp.columns)

    # Per-ticker importance-over-months matrix: CORN has 2 OOS months → 2 columns.
    mat = pd.read_csv(
        batch / "per_ticker" / "CORN" /
        "importance_over_months__gpt4omini120650__gradient_boosting.csv", index_col=0)
    assert list(mat.columns) == ["2024-03", "2024-04"]
    assert mat.shape[0] == 4  # four factors

    assert (batch / "per_ticker" / "CORN" /
            "importance_over_months__gpt4omini120650__gradient_boosting.png").exists()
    assert any((batch / "per_ticker" / "CORN").glob("performance_*.png"))
    assert (batch / "summary.md").exists()
    assert "summary" in paths


def test_aggregate_empty_batch_is_safe(tmp_path):
    batch = tmp_path / "empty"
    (batch / "runs").mkdir(parents=True)
    paths = rolling.aggregate(batch)  # no runs → no combined tables, summary still written
    assert (batch / "summary.md").exists()
    assert "bruteforce_all" not in paths
