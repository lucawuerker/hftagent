"""Tests for the IndNeutralize / market-cap formulaic alphas.

Covers the 15 alphas unblocked by the FMP fundamental data (#63, #67,
#69, #70, #76, #79, #80, #82, #87, #89, #90, #91, #93, #97, #100) plus
the four earlier fallback implementations (#48, #56, #58, #59):

- registry discovery picks every one up under the ``alpha_NNN`` id;
- ``calc`` runs on a synthetic panel with object-dtype label frames and
  a daily ``marketCap`` frame, returns an aligned, not-all-NaN frame;
- the IndNeutralize path demeans within groups and actually changes the
  output;
- a missing ``subindustry`` field falls back to ``industry``;
- Alpha#56 prefers ``marketCap`` over legacy ``cap`` over the volume
  proxy.

Run: ``./venv/bin/python -m pytest tests/test_formulaic_alphas_fundamental.py``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.factors._discover import discover_factors
from quant_fund_agent.factors.ops import indneutralize
from quant_fund_agent.factors.registry import get_all_factor_classes, instantiate_factor

discover_factors()

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
N_BARS = 320  # Alpha#63 needs ~240 bars (adv180 → sum 37 → corr 14 → decay 12)

SECTORS = {
    "AAA": "Technology", "BBB": "Technology", "CCC": "Technology",
    "DDD": "Energy", "EEE": "Energy", "FFF": "Energy",
}
INDUSTRIES = {
    "AAA": "Software", "BBB": "Software", "CCC": "Hardware",
    "DDD": "Oil", "EEE": "Oil", "FFF": "Solar",
}
SUBINDUSTRIES = {
    "AAA": "AppSoftware", "BBB": "AppSoftware", "CCC": "PC",
    "DDD": "OilGas", "EEE": "OilGas", "FFF": "SolarPanels",
}

NEW_ALPHA_IDS = [
    "alpha_063", "alpha_067", "alpha_069", "alpha_070", "alpha_076",
    "alpha_079", "alpha_080", "alpha_082", "alpha_087", "alpha_089",
    "alpha_090", "alpha_091", "alpha_093", "alpha_097", "alpha_100",
]
UPGRADED_ALPHA_IDS = ["alpha_048", "alpha_056", "alpha_058", "alpha_059"]


def _label_frame(index: pd.DatetimeIndex, mapping: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {t: [mapping[t]] * len(index) for t in TICKERS}, index=index, dtype=object
    )


def build_panel(seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=N_BARS)

    close = pd.DataFrame(
        {
            t: (50.0 + 10.0 * i)
            * np.cumprod(1.0 + 0.01 * rng.standard_normal(N_BARS))
            for i, t in enumerate(TICKERS)
        },
        index=idx,
    )
    open_ = close * (1.0 + 0.002 * rng.standard_normal(close.shape))
    spread = np.abs(0.005 * rng.standard_normal(close.shape)) + 0.001
    high = pd.DataFrame(
        np.maximum(open_.to_numpy(), close.to_numpy()) * (1.0 + spread),
        index=idx, columns=TICKERS,
    )
    low = pd.DataFrame(
        np.minimum(open_.to_numpy(), close.to_numpy()) * (1.0 - spread),
        index=idx, columns=TICKERS,
    )
    volume = pd.DataFrame(
        np.exp(rng.normal(13.0, 0.4, close.shape)), index=idx, columns=TICKERS
    )
    shares = {t: 1e9 * (i + 1) for i, t in enumerate(TICKERS)}
    market_cap = close * pd.Series(shares)

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "marketCap": market_cap,
        "sector": _label_frame(idx, SECTORS),
        "industry": _label_frame(idx, INDUSTRIES),
        "subindustry": _label_frame(idx, SUBINDUSTRIES),
    }


@pytest.fixture(scope="module")
def panel() -> dict[str, pd.DataFrame]:
    return build_panel()


# ── registry / discovery ─────────────────────────────────────────────────────

def test_all_alphas_registered():
    registry = get_all_factor_classes()
    for fid in NEW_ALPHA_IDS + UPGRADED_ALPHA_IDS:
        assert fid in registry, f"{fid} missing from factor registry"


def test_new_alphas_declare_label_inputs():
    registry = get_all_factor_classes()
    label_fields = {"sector", "industry", "subindustry"}
    for fid in NEW_ALPHA_IDS:
        assert label_fields & set(registry[fid].inputs), (
            f"{fid} should declare its IndNeutralize label field in `inputs`"
        )


# ── calc runs, aligned, not all-NaN ──────────────────────────────────────────

@pytest.mark.parametrize("fid", NEW_ALPHA_IDS + UPGRADED_ALPHA_IDS)
def test_calc_runs_aligned_not_all_nan(fid, panel):
    result = instantiate_factor(fid).calc(panel)
    assert isinstance(result, pd.DataFrame)
    assert result.index.equals(panel["close"].index)
    assert list(result.columns) == TICKERS
    assert result.notna().any().any(), f"{fid} produced an all-NaN signal"


# ── IndNeutralize semantics ──────────────────────────────────────────────────

def test_indneutralize_demeans_within_groups(panel):
    """Group means are ~0 after neutralising by the alphas' label frames."""
    neut = indneutralize(panel["close"], panel["industry"])
    for label in set(INDUSTRIES.values()):
        cols = [t for t in TICKERS if INDUSTRIES[t] == label]
        group_mean = neut[cols].mean(axis=1)
        assert np.nanmax(np.abs(group_mean.to_numpy())) < 1e-8, (
            f"industry group {label!r} not demeaned"
        )


def test_neutralization_changes_alpha_output(panel):
    """Alpha#63 with labels differs from the skip-neutralisation path."""
    alpha = instantiate_factor("alpha_063")
    with_labels = alpha.calc(panel)
    stripped = {
        k: v for k, v in panel.items()
        if k not in ("sector", "industry", "subindustry")
    }
    without_labels = alpha.calc(stripped)
    assert not np.allclose(
        with_labels.to_numpy(), without_labels.to_numpy(), equal_nan=True
    ), "IndNeutralize path had no effect on Alpha#63"


# ── subindustry → industry fallback ──────────────────────────────────────────

@pytest.mark.parametrize("fid", ["alpha_067", "alpha_090", "alpha_100"])
def test_missing_subindustry_falls_back_to_industry(fid, panel):
    alpha = instantiate_factor(fid)

    no_sub = {k: v for k, v in panel.items() if k != "subindustry"}
    fallback = alpha.calc(no_sub)
    assert fallback.notna().any().any(), f"{fid} all-NaN under fallback"

    sub_as_industry = dict(panel)
    sub_as_industry["subindustry"] = panel["industry"]
    expected = alpha.calc(sub_as_industry)

    pd.testing.assert_frame_equal(fallback, expected)


# ── Alpha#56 market-cap preference ───────────────────────────────────────────

def test_alpha056_prefers_marketcap_over_cap_and_proxy(panel):
    alpha = instantiate_factor("alpha_056")

    with_mc = alpha.calc(panel)

    # marketCap wins even when a (different) legacy cap frame is present.
    both = dict(panel)
    both["cap"] = panel["volume"] * 3.0
    assert np.allclose(
        alpha.calc(both).to_numpy(), with_mc.to_numpy(), equal_nan=True
    )

    # legacy cap is used when marketCap is absent.
    cap_only = {k: v for k, v in panel.items() if k != "marketCap"}
    cap_only["cap"] = panel["marketCap"]
    assert np.allclose(
        alpha.calc(cap_only).to_numpy(), with_mc.to_numpy(), equal_nan=True
    )

    # volume proxy differs from the real-cap result.
    proxy = {k: v for k, v in panel.items() if k != "marketCap"}
    without_mc = alpha.calc(proxy)
    assert not np.allclose(
        with_mc.to_numpy(), without_mc.to_numpy(), equal_nan=True
    ), "marketCap upgrade had no effect on Alpha#56"
