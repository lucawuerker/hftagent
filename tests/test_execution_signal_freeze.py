"""Tests for the FrozenSignalSet interface artifact (E0).

Covers: versioned manifest + parquet round-trip, IS-only fitting, the
poison audit (a leak in the fit MUST be caught), and default spec diversity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.execution.signal_freeze import (
    FrozenSignalSet,
    default_specs,
    freeze_eval_signals,
)
from quant_fund_agent.research_eval.splits import ThreeWaySplit, three_way_split

N_BARS, TICKERS = 480, ["A", "B", "C", "D", "E", "F"]


def _factor_code(fid: str, body: str) -> str:
    return f'''\
"""Test factor {fid}."""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import stddev, ts_mean, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class F_{fid}(BaseFactor):
    factor_id = "{fid}"
    name = "{fid}"
    category = "momentum"
    inputs = ["close"]
    prediction_horizon = 1

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        return {body}
'''


BOOK = [
    {"factor_id": "frz_mom", "code": _factor_code("frz_mom", "close.pct_change().fillna(0.0)")},
    {"factor_id": "frz_rank",
     "code": _factor_code("frz_rank", "(ts_rank(close, 8) - 0.5).fillna(0.0)")},
]


@pytest.fixture()
def dev_panel_and_split():
    """AR(1) returns → real momentum edge; panel is already dev-sliced (IS∪VAL)."""
    rng = np.random.default_rng(3)
    rets = np.zeros((N_BARS, len(TICKERS)))
    eps = rng.standard_normal((N_BARS, len(TICKERS))) * 0.01
    for t in range(1, N_BARS):
        rets[t] = 0.6 * rets[t - 1] + eps[t]
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="D")
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=TICKERS)
    full = three_way_split(idx, is_frac=0.6, val_frac=0.2)
    dev = full.is_val_mask
    panel_dev = {"close": close.loc[dev]}
    split = ThreeWaySplit(is_mask=full.is_mask[dev], val_mask=full.val_mask[dev],
                          test_mask=full.test_mask[dev])
    return panel_dev, split


SPECS = [{"model": "ridge", "subset": [0, 1]}, {"model": "ridge", "subset": [0]}]


def test_freeze_roundtrip_and_manifest(tmp_path, dev_panel_and_split):
    panel, split = dev_panel_and_split
    fs = freeze_eval_signals(BOOK, panel, split, out_dir=tmp_path, version=1,
                             target_horizon=1, specs=SPECS)
    assert fs.version == 1
    assert fs.k == 2
    assert fs.manifest_path.exists()
    assert fs.manifest["book_ids"] == ["frz_mom", "frz_rank"]
    assert fs.manifest["signals"][0]["subset"] == ["frz_mom", "frz_rank"]

    reloaded = FrozenSignalSet.from_manifest(fs.manifest_path)
    for a, b in zip(fs.load(), reloaded.load()):
        pd.testing.assert_frame_equal(a, b)
    # frames live on the DEV grid only — TEST rows physically absent
    assert len(reloaded.load()[0]) == len(panel["close"])


def test_freeze_versioning_creates_separate_dirs(tmp_path, dev_panel_and_split):
    panel, split = dev_panel_and_split
    v1 = freeze_eval_signals(BOOK, panel, split, out_dir=tmp_path, version=1,
                             target_horizon=1, specs=SPECS)
    v2 = freeze_eval_signals(BOOK[:1], panel, split, out_dir=tmp_path, version=2,
                             target_horizon=1, specs=[{"model": "ridge", "subset": [0]}])
    assert v1.directory != v2.directory
    assert v1.directory.name == "v1" and v2.directory.name == "v2"
    assert FrozenSignalSet.from_manifest(v1.manifest_path).manifest["book_hash"] \
        != FrozenSignalSet.from_manifest(v2.manifest_path).manifest["book_hash"]


def test_poison_audit_passes_for_clean_fit(tmp_path, dev_panel_and_split):
    panel, split = dev_panel_and_split
    fs = freeze_eval_signals(BOOK, panel, split, out_dir=tmp_path, version=1,
                             target_horizon=1, specs=SPECS)
    audit = fs.poison_audit()
    assert audit["audited"] is True
    assert audit["passed"] is True
    assert all(a["max_is_row_diff"] == 0.0 for a in audit["per_signal"])


def test_poison_audit_catches_val_leaking_fit(tmp_path, dev_panel_and_split,
                                              monkeypatch):
    """Break the IS-only discipline → the audit MUST fail (the leak proof)."""
    from quant_fund_agent.execution import signal_freeze as sf

    real_combined = sf._combined

    def leaky_combined(signals, panel, is_mask, model, target_horizon):
        leaked = np.ones(len(is_mask), dtype=bool)   # fit on ALL rows incl. VAL
        return real_combined(signals, panel, leaked, model, target_horizon)

    monkeypatch.setattr(sf, "_combined", leaky_combined)
    fs = freeze_eval_signals(BOOK, panel := dev_panel_and_split[0],
                             dev_panel_and_split[1], out_dir=tmp_path, version=1,
                             target_horizon=1, specs=SPECS)
    assert fs.poison_audit()["passed"] is False


def test_default_specs_are_diverse():
    specs = default_specs(4)
    assert len(specs) == 4
    models = {s["model"] for s in specs}
    subsets = [tuple(s["subset"]) for s in specs]
    assert len(models) == 2               # two model families
    assert len(set(subsets)) >= 3         # and different factor subsets
    assert default_specs(1) == [{"model": "ridge", "subset": [0]},
                                {"model": "gradient_boosting", "subset": [0]}]


def test_empty_book_rejected(tmp_path, dev_panel_and_split):
    panel, split = dev_panel_and_split
    with pytest.raises(ValueError, match="empty book"):
        freeze_eval_signals([], panel, split, out_dir=tmp_path)
