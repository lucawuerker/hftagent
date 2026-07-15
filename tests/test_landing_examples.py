"""Tests for the landing-page example generator (showcase_pipeline/landing_examples)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.workspace import Scope
from showcase_pipeline.landing_examples.export import export_example, rebuild_index
from showcase_pipeline.landing_examples.metrics import (
    CardMetrics,
    assemble_card_metrics,
    per_strategy_pbo,
    svg_points,
)
from showcase_pipeline.landing_examples.story import build_story_md
from showcase_pipeline.landing_examples.transcript import build_transcript
from showcase_pipeline.landing_examples.verdict import (
    BADGE_OVERFIT,
    BADGE_ROBUST,
    BADGE_WORTH_TESTING,
    ComplianceError,
    assign_badge,
    check_compliance,
    verdict_note,
)

TIMING_DIAG = Path("data/workspaces/yfinance_equity_sp100/preruns/timing_diag")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _serialise(s: pd.Series) -> dict:
    return {"timestamps": [ts.isoformat() for ts in s.index],
            "values": [float(v) for v in s.values]}


def _returns(seed: int, n: int = 240, mean: float = 0.0, std: float = 0.01,
             start: str = "2023-01-02") -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(mean + std * rng.standard_normal(n), index=idx)


def _trial(i: int, sharpe: float, returns: pd.Series) -> dict:
    return {
        "iteration": i,
        "spec": {"model_type": "ridge", "factor_ids": ["f_a", "f_b"],
                 "strategy_name": f"trial {i}"},
        "metrics": {"sharpe_ratio": sharpe,
                    "portfolio_returns": _serialise(returns)},
    }


def _candidate(dsr: float = 0.31, dsr_verdict: str = "fail",
               oos_verdict: str = "fail", oos_sharpe: float = -0.2,
               is_sharpe: float = 2.1, n_trials: int = 3) -> dict:
    trials = [_trial(i, 1.0 + 0.3 * i, _returns(seed=i)) for i in range(n_trials)]
    final_returns = _returns(seed=99, mean=0.0005)
    return {
        "attempt": 4,
        "approved": False,
        "reject_stage": "statistician",
        "strategy_id": None,
        "hypothesis": "Combine open-close spread reversion with volume dynamics "
                      "to fade one-day overreactions in large caps.",
        "selection_rationale": "Both factors show complementary decay profiles.",
        "spec": {
            "strategy_name": "Open-close reversion blend",
            "model_type": "static_weights",
            "weights": {"open_close_spread": 1.0},
            "factor_ids": [],
            "target_horizon": 6,
            "holding_period": 6,
            "max_positions": 5,
            "equal_weight": False,
            "min_conviction": 0.0,
            "position_construction": "cross_sectional",
            "position_params": {},
            "model_artifact_path": "",
            "model_params": {},
            "reasoning": "",
        },
        "backtest_metrics": {
            "sharpe_ratio": is_sharpe,
            "portfolio_returns": _serialise(final_returns),
            "equity_curve": _serialise((1 + final_returns).cumprod() - 1),
        },
        "trial_history": trials,
        "stat_result": {
            "final_decision": "reject",
            "final_reasoning": "",
            "test_results": [
                {"test_id": "deflated_sharpe_ratio", "verdict": dsr_verdict,
                 "value": dsr,
                 "details": {"n_trials": n_trials,
                             "expected_max_sharpe_annualised": 1.5,
                             "sharpe_annualised": is_sharpe}},
                {"test_id": "out_of_sample_backtest", "verdict": oos_verdict,
                 "value": oos_sharpe,
                 "details": {"is_sharpe": is_sharpe, "oos_sharpe": oos_sharpe}},
            ],
        },
        "oos_split_ratio": 0.2,
        "universe": {"provider": "yfinance", "asset_class": "equity",
                     "universe": "sp100", "frequency": "1d",
                     "timespan": ["2020-01-01", "2026-06-01"]},
    }


def _curves(is_returns: pd.Series, oos_returns: pd.Series) -> dict:
    is_curve = (1 + is_returns).cumprod() - 1
    is_end = float(is_curve.iloc[-1])
    oos_curve = (1 + is_end) * ((1 + oos_returns).cumprod()) - 1
    return {"is_curve": is_curve, "oos_curve": oos_curve,
            "is_returns": is_returns, "oos_returns": oos_returns,
            "oos_sharpe_recomputed": -0.21, "oos_sharpe_recorded": -0.2,
            "recompute_match": True}


# ---------------------------------------------------------------------------
# per-strategy PBO
# ---------------------------------------------------------------------------

class TestPerStrategyPBO:
    def test_noise_vs_genuine_edge(self):
        noise_trials = [_trial(i, 0.5, _returns(seed=10 + i)) for i in range(5)]
        edge_returns = _returns(seed=42, mean=0.002, std=0.005)
        edge_trials = [_trial(0, 3.0, edge_returns)] + [
            _trial(i, 0.1, _returns(seed=20 + i, std=0.02)) for i in range(1, 5)]

        noise = per_strategy_pbo(noise_trials)
        edge = per_strategy_pbo(edge_trials)
        assert noise["pbo"] is not None and 0.0 <= noise["pbo"] <= 1.0
        assert edge["pbo"] is not None
        # A dominant genuine edge keeps winning out-of-sample.
        assert edge["pbo"] < noise["pbo"]
        assert edge["pbo"] <= 0.1

    def test_single_trial_undefined(self):
        out = per_strategy_pbo([_trial(0, 1.0, _returns(seed=1))])
        assert out["pbo"] is None and out["n_splits"] == 0

    def test_no_series_undefined(self):
        trials = [{"iteration": 0, "spec": {}, "metrics": {"sharpe_ratio": 1.0}}]
        assert per_strategy_pbo(trials)["pbo"] is None

    def test_misaligned_indices_inner_join(self):
        a = _returns(seed=1, n=200, start="2023-01-02")
        b = _returns(seed=2, n=200, start="2023-02-01")
        out = per_strategy_pbo([_trial(0, 1.0, a), _trial(1, 1.0, b)])
        # Overlap is still long enough for 8 groups → defined.
        assert out["pbo"] is not None

    def test_too_short_overlap_undefined(self):
        a = _returns(seed=1, n=5)
        b = _returns(seed=2, n=5)
        assert per_strategy_pbo([_trial(0, 1.0, a), _trial(1, 1.0, b)])["pbo"] is None


# ---------------------------------------------------------------------------
# badge + verdict copy
# ---------------------------------------------------------------------------

def _cm(pbo, dsr, oos) -> CardMetrics:
    return CardMetrics(strategy_name="s", is_sharpe=2.0, dsr_prob=dsr,
                       oos_verdict=oos, oos_sharpe=0.8, oos_is_sharpe=2.0,
                       pbo=pbo, pbo_n_splits=70, n_trials=6)


class TestBadge:
    @pytest.mark.parametrize("pbo,dsr,oos,expected", [
        (0.10, 0.80, "pass", BADGE_ROBUST),
        (0.25, 0.75, "pass", BADGE_ROBUST),          # boundary inclusive
        (0.30, 0.80, "pass", BADGE_WORTH_TESTING),   # pbo too high for robust
        (0.10, 0.70, "pass", BADGE_WORTH_TESTING),   # dsr below robust bar
        (0.10, 0.80, "warn", BADGE_WORTH_TESTING),   # oos not clean
        (0.50, 0.80, "pass", BADGE_OVERFIT),         # pbo at overfit bar
        (0.10, 0.59, "pass", BADGE_OVERFIT),         # dsr below warn floor
        (0.10, 0.80, "fail", BADGE_OVERFIT),         # oos failed
        (None, 0.90, "pass", BADGE_WORTH_TESTING),   # no pbo → never robust
        (None, 0.30, "pass", BADGE_OVERFIT),
    ])
    def test_truth_table(self, pbo, dsr, oos, expected):
        assert assign_badge(_cm(pbo, dsr, oos)) == expected


class TestVerdictCopy:
    def test_notes_contain_numbers_and_pass_compliance(self):
        for badge_case in [(0.81, 0.22, "fail", BADGE_OVERFIT),
                           (0.12, 0.90, "pass", BADGE_ROBUST),
                           (0.35, 0.70, "warn", BADGE_WORTH_TESTING)]:
            pbo, dsr, oos, badge = badge_case
            m = _cm(pbo, dsr, oos)
            note = verdict_note(m, badge)
            assert f"{pbo:.0%}" in note
            check_compliance(note)  # must not raise

    def test_banned_words_raise(self):
        for text in ["This is guaranteed to work", "a RISK-FREE edge",
                     "we beat the market", "proven returns here"]:
            with pytest.raises(ComplianceError):
                check_compliance(text)


# ---------------------------------------------------------------------------
# metrics assembly
# ---------------------------------------------------------------------------

class TestAssemble:
    def test_full_candidate(self):
        m = assemble_card_metrics(_candidate())
        assert m.strategy_name == "Open-close reversion blend"
        assert m.is_sharpe == 2.1
        assert m.dsr_prob == 0.31
        assert m.oos_verdict == "fail"
        assert m.n_trials == 3
        assert m.card_ready
        assert m.pbo is not None  # computed from the 3 trials' series
        assert assign_badge(m) == BADGE_OVERFIT

    def test_architect_reject_not_card_ready(self):
        c = _candidate()
        c["stat_result"] = None
        c["reject_stage"] = "architect"
        m = assemble_card_metrics(c)
        assert not m.card_ready


# ---------------------------------------------------------------------------
# SVG scaling
# ---------------------------------------------------------------------------

class TestSvgPoints:
    def test_points_within_viewbox_and_stitched(self):
        is_returns = _returns(seed=5, mean=0.001)
        oos_returns = _returns(seed=6, mean=-0.0005, n=60, start="2023-12-01")
        c = _curves(is_returns, oos_returns)
        pts = svg_points(c["is_curve"], c["oos_curve"])
        for key, x_lo, x_hi in [("points_is", 20, 340), ("points_oos", 340, 580)]:
            pairs = [tuple(map(float, p.split(","))) for p in pts[key].split()]
            for x, y in pairs:
                assert x_lo <= x <= x_hi
                assert 30 <= y <= 235
        # OOS polyline starts exactly at the IS polyline's final point.
        assert pts["points_oos"].split()[0] == pts["points_is"].split()[-1]


# ---------------------------------------------------------------------------
# transcript
# ---------------------------------------------------------------------------

class TestTranscript:
    def test_grounded_and_compliant(self):
        c = _candidate()
        m = assemble_card_metrics(c)
        badge = assign_badge(m)
        note = verdict_note(m, badge)
        t = build_transcript(c, m, badge, note)
        assert t["badge"] == BADGE_OVERFIT
        assert t["llm_polished"] is False
        roles = [msg["role"] for msg in t["messages"]]
        assert roles[0] == "user" and "progress" in roles
        # The verdict message carries the deterministic note verbatim.
        assert t["messages"][-1]["text"] == note


# ---------------------------------------------------------------------------
# export (into tmp_path — never touches company-brain)
# ---------------------------------------------------------------------------

class TestExport:
    def test_full_data_pack(self, tmp_path):
        c = _candidate()
        curves = _curves(_returns(seed=5, mean=0.001),
                         _returns(seed=6, mean=-0.0005, n=60, start="2023-12-01"))
        scope = Scope("testcfg", "testprerun", root=tmp_path / "ws")
        out = export_example(c, "overfit-example", tmp_path / "examples",
                             scope, curves=curves)

        for fname in ["card.json", "equity_curve.json", "equity_curve.csv",
                      "card.png", "behind_the_verdict.md",
                      "chat_transcript.json", "provenance.json"]:
            assert (out / fname).exists(), fname

        card = json.loads((out / "card.json").read_text())
        assert card["badge"] == BADGE_OVERFIT
        assert card["caveat"]
        assert card["stats"]["pbo_pct"] is not None
        assert card["stats"]["n_trials"] == 3

        eq = json.loads((out / "equity_curve.json").read_text())
        assert eq["viewBox"] == "0 0 600 260"
        # Stitch: OOS series starts from the IS terminal value.
        assert eq["oos"][0][1] == pytest.approx(
            (1 + eq["is"][-1][1]) * (1 + curves["oos_returns"].iloc[0]) - 1,
            rel=1e-3)

        prov = json.loads((out / "provenance.json").read_text())
        assert prov["attempt"] == 4
        assert prov["recompute_match"] is True
        assert prov["llm_polished_transcript"] is False

        index = rebuild_index(tmp_path / "examples")
        assert "overfit-example" in index.read_text()

    def test_architect_reject_refused(self, tmp_path):
        c = _candidate()
        c["stat_result"] = None
        scope = Scope("testcfg", "testprerun", root=tmp_path / "ws")
        with pytest.raises(ValueError, match="not card-ready"):
            export_example(c, "x", tmp_path / "examples", scope,
                           curves=_curves(_returns(seed=1), _returns(seed=2)))


# ---------------------------------------------------------------------------
# story (against the real read-only timing_diag prerun when present)
# ---------------------------------------------------------------------------

class TestStory:
    def test_story_md_synthetic_scope(self, tmp_path):
        c = _candidate()
        m = assemble_card_metrics(c)
        badge = assign_badge(m)
        note = verdict_note(m, badge)
        scope = Scope("testcfg", "testprerun", root=tmp_path / "ws")
        md = build_story_md(c, m, badge, note, scope)
        assert "# Behind the verdict" in md
        assert badge in md
        assert "probability of backtest overfitting" in md.lower()
        assert "not investment advice" in md.lower()

    @pytest.mark.skipif(not TIMING_DIAG.exists(),
                        reason="timing_diag prerun not on disk")
    def test_story_md_real_prerun(self):
        c = _candidate()
        # Point the card at a real researched factor of the prerun.
        c["spec"]["weights"] = {"open_close_spread": 1.0}
        m = assemble_card_metrics(c)
        badge = assign_badge(m)
        note = verdict_note(m, badge)
        scope = Scope("yfinance_equity_sp100", "timing_diag")
        md = build_story_md(c, m, badge, note, scope)
        assert "open_close_spread" in md
        # Trading idea + evolution provenance resolved from the real artifacts.
        assert "Trading idea" in md
