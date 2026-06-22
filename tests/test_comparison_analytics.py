"""Unit tests for the factor-analytics comparison track + speed/reliability seams.

These run on a tiny synthetic panel (no LOBSTER on disk): we pre-populate the
modeling-service signal cache so the analytics functions read our crafted signals
directly, exercising the maths (diversity/redundancy, IC deflation, model-based
importance) without instantiating any factor classes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.comparison import analytics, factors, report
from quant_fund_agent.comparison.config import ComparisonConfig
from quant_fund_agent.modeling import service


N_BARS, N_TICKERS = 300, 8
TICKERS = [f"T{i}" for i in range(N_TICKERS)]


def _frame(values: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="min")
    return pd.DataFrame(values, index=idx, columns=TICKERS)


@pytest.fixture
def synthetic(monkeypatch):
    """A panel with 3 near-duplicate factors + 3 independent ones in the cache."""
    rng = np.random.default_rng(0)
    close = _frame(100 + rng.standard_normal((N_BARS, N_TICKERS)).cumsum(axis=0))

    base = rng.standard_normal((N_BARS, N_TICKERS))
    signals = {
        "dup_a": _frame(base),
        "dup_b": _frame(base + 0.01 * rng.standard_normal((N_BARS, N_TICKERS))),
        "dup_c": _frame(base + 0.01 * rng.standard_normal((N_BARS, N_TICKERS))),
        "ind_x": _frame(rng.standard_normal((N_BARS, N_TICKERS))),
        "ind_y": _frame(rng.standard_normal((N_BARS, N_TICKERS))),
        "ind_z": _frame(rng.standard_normal((N_BARS, N_TICKERS))),
    }
    # Inject directly into the cache so _factor_signal returns them as-is.
    monkeypatch.setattr(service, "_SIGNAL_CACHE", dict(signals))
    panel = {"close": close}
    cfg = ComparisonConfig(preruns=["z"], target_horizon=1, analytics_max_rows=10_000,
                           corr_threshold=0.7, seed=0, importance_models=("lasso", "gradient_boosting"))
    return panel, cfg, list(signals)


def test_diversity_detects_redundancy(synthetic):
    panel, cfg, fids = synthetic
    summary, factor_rows, corr = analytics.evaluate_prerun_diversity("z", fids, panel, cfg)

    # 3 near-duplicates + 3 independents → the duplicates collapse into one
    # dominant variance direction, so the (variance-weighted) effective count is
    # ~3, well below the raw 6.
    assert summary["n_factors"] == 6
    assert 2.5 < summary["eff_n_factors"] < 4.5
    assert summary["eff_ratio"] < 1.0
    assert summary["redundancy"] > 0.0
    # The duplicate cluster collapses: 3 indep + 1 dup-cluster = 4 clusters.
    assert summary["n_clusters"] == 4
    # Duplicates are near-perfectly correlated to a sibling.
    by_id = {r["factor_id"]: r for r in factor_rows}
    assert by_id["dup_a"]["max_abs_corr"] > 0.95
    assert by_id["ind_x"]["max_abs_corr"] < 0.5
    assert corr.shape == (6, 6)


def test_deflation_haircuts_and_grows_with_breadth(synthetic):
    _, cfg, _ = synthetic
    # Same best IC, different number of factors tried → bigger k haircuts more.
    def rows(k: int):
        out = [{"prerun": "z", "factor_id": f"f{i}", "ic_1": 0.05, "n_timestamps": 1000}
               for i in range(k)]
        out[0]["ic_1"] = 0.08  # the "best" factor
        return out

    small = analytics._deflate_ic("z", [f"f{i}" for i in range(5)], rows(5), cfg)
    large = analytics._deflate_ic("z", [f"f{i}" for i in range(80)], rows(80), cfg)

    assert small["best_ic"] == pytest.approx(0.08)
    assert small["deflated_best_ic"] < small["best_ic"]          # haircut applied
    assert large["deflated_best_ic"] < small["deflated_best_ic"]  # more trials → more haircut
    assert large["ic_n_tested"] == 80


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_modelview_importance_returns_nonzero_subset(synthetic):
    panel, cfg, fids = synthetic
    ic_rows = [{"prerun": "z", "factor_id": f, "ic_1": 0.03, "n_timestamps": 800} for f in fids]
    summary, importance_rows = analytics.evaluate_prerun_modelview(
        "z", fids, panel, cfg, ic_rows=ic_rows)

    assert importance_rows, "expected per-factor importance rows"
    assert {r["model"] for r in importance_rows} <= {"lasso", "gradient_boosting"}
    assert all(0.0 <= r["importance"] <= 1.0 for r in importance_rows)
    # LASSO sparsity is reported and well-formed.
    assert 0 <= summary["lasso_n_nonzero"] <= len(fids)


def test_subsample_bars_caps_every_field():
    a = _frame(np.zeros((N_BARS, N_TICKERS)))
    b = _frame(np.ones((N_BARS, N_TICKERS)))
    out = factors._subsample_bars({"close": a, "volume": b}, max_bars=50)
    assert len(out["close"]) <= 50
    assert out["close"].index.equals(out["volume"].index)  # aligned across fields
    # No-op when already small enough.
    same = factors._subsample_bars({"close": a}, max_bars=10_000)
    assert len(same["close"]) == N_BARS


def test_checkpoint_persists_ic_before_later_tracks(tmp_path):
    """write_tables (the checkpoint primitive) persists IC mid-run."""
    cfg = ComparisonConfig(preruns=["z"], output_root=str(tmp_path), comparison_id="ckpt")
    results = {"usability": {"z": {"n_usable": 1, "n_dropped": 0}},
               "ic_rows": [{"prerun": "z", "factor_id": "f0", "ic_1": 0.04}]}
    report.write_tables(cfg, results)
    assert (tmp_path / "ckpt" / "ic_results.csv").exists()
    assert (tmp_path / "ckpt" / "usability.json").exists()
