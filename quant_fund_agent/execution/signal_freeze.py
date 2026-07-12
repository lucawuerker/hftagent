"""Freeze the K evaluation signals — the executor arm's interface artifact.

The cross-signal generalisation axis scores every executor candidate against
the **same K frozen composite signals** (different model families × different
factor subsets), so no candidate can co-adapt to one alpha.  Freezing is a
first-class, versioned artifact (DESIGN §Overfitting, revised 2026-07-11):

* ``freeze_eval_signals`` fits each spec's combined model **on IS only** (with
  the label-availability discipline of ``research_eval.harness``), predicts
  over the development (IS∪VAL) grid, and persists the frames as parquet under
  ``<out_dir>/frozen_signals/v<k>/`` next to a ``manifest.json`` recording the
  book hash, specs, panel window key, split sizes and the poison-audit result.
* A standalone execution-evolution run freezes once at run start (``v1``); the
  joint outer layer re-freezes at factor-block boundaries (``v2``, …) and
  re-scores the executor archive (see ``docs/joint-evolution/DESIGN.md``).
* **Poison audit** (leak proof, recorded per signal): corrupt every VALIDATION
  row of the panel and refit — the model's predictions on **IS rows** must be
  bit-identical, proving the fit consumed no VAL/TEST information (features at
  VAL rows may of course change; they are causal per-row inputs).

A leaky frozen signal would launder look-ahead into *every* executor score —
the worst leak available at this layer — which is why the audit runs at freeze
time, not review time.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("execution.signal_freeze")

#: default spec set: 2 model families × (all, front-half, back-half) subsets → K=4
DEFAULT_MODELS = ("ridge", "gradient_boosting")


def default_specs(n_book: int, models: Sequence[str] = DEFAULT_MODELS) -> list[dict]:
    """K diverse (model, factor-subset) specs from a book of ``n_book`` factors.

    Diversity per DESIGN locked decision 6: different model families AND
    different factor subsets.  With ≥ 2 factors: each model on the full book,
    plus the first model on the front half and the second on the back half
    (K = 4).  A 1-factor book degenerates to one spec per model (K = 2).
    """
    models = list(models)
    specs: list[dict] = [
        {"model": m, "subset": list(range(n_book))} for m in models
    ]
    if n_book >= 2:
        half = n_book // 2
        specs.append({"model": models[0], "subset": list(range(half))})
        specs.append({"model": models[-1], "subset": list(range(half, n_book))})
    return specs


def _book_hash(book: Sequence[dict[str, Any]]) -> str:
    payload = "\n".join(
        f"{p.get('factor_id')}:{''.join(str(p.get('code', '')).split())}"
        for p in book
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _window_key(close: pd.DataFrame) -> list:
    idx = close.index
    if len(idx) == 0:
        return [0, None, None, len(close.columns)]
    return [len(idx), str(idx[0]), str(idx[-1]), len(close.columns)]


@dataclass
class FrozenSignalSet:
    """A versioned bundle of K frozen evaluation-signal frames + its manifest."""

    directory: Path
    manifest: dict[str, Any]
    _signals: list[pd.DataFrame] | None = field(default=None, repr=False)

    @property
    def version(self) -> int:
        return int(self.manifest.get("version", 1))

    @property
    def manifest_path(self) -> Path:
        return self.directory / "manifest.json"

    @property
    def k(self) -> int:
        return len(self.manifest.get("signals", []))

    def load(self) -> list[pd.DataFrame]:
        """The K frozen frames (cached after first read)."""
        if self._signals is None:
            self._signals = [
                pd.read_parquet(self.directory / entry["file"])
                for entry in self.manifest.get("signals", [])
            ]
        return self._signals

    def poison_audit(self) -> dict[str, Any]:
        return self.manifest.get("poison_audit", {})

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "FrozenSignalSet":
        p = Path(manifest_path)
        return cls(directory=p.parent, manifest=json.loads(p.read_text()))


def _compiled_signals(book: Sequence[dict[str, Any]],
                      panel: dict[str, Any]) -> list[pd.DataFrame]:
    """In-memory compile every book program and compute its signal on ``panel``."""
    from quant_fund_agent.factors.inmem import signal_from_code

    return [signal_from_code(p["code"], p["factor_id"], panel) for p in book]


def _combined(signals: Sequence[pd.DataFrame], panel: dict[str, Any],
              is_mask: np.ndarray, model: str, target_horizon: int) -> pd.DataFrame | None:
    from quant_fund_agent.comparison.config import ComparisonConfig
    from quant_fund_agent.research_eval.harness import _combined_prediction

    cfg = ComparisonConfig(preruns=["freeze"], target_horizon=target_horizon,
                           fit_standardize="per_underlying", seed=0)
    return _combined_prediction(list(signals), panel["close"], is_mask, cfg, model)


def freeze_eval_signals(
    book: Sequence[dict[str, Any]],
    panel: dict[str, Any],
    split: Any,
    *,
    out_dir: str | Path,
    version: int = 1,
    target_horizon: int = 6,
    specs: Sequence[dict[str, Any]] | None = None,
    audit: bool = True,
) -> FrozenSignalSet:
    """Materialise the K frozen evaluation signals from a factor book.

    ``book`` is a list of ``{"factor_id", "code"}`` programs; ``panel`` must
    already be **dev-sliced** (IS∪VAL only — the caller/service guarantees TEST
    is physically absent, the dev-slice convention); ``split`` is the
    dev-relative :class:`~quant_fund_agent.research_eval.splits.ThreeWaySplit`.
    Fits are IS-only.  Writes ``<out_dir>/frozen_signals/v<version>/`` and
    returns the :class:`FrozenSignalSet`.
    """
    if not book:
        raise ValueError("cannot freeze evaluation signals from an empty book")
    close = panel["close"]
    is_mask = np.asarray(split.is_mask, dtype=bool)
    specs = list(specs) if specs is not None else default_specs(len(book))

    directory = Path(out_dir) / "frozen_signals" / f"v{int(version)}"
    directory.mkdir(parents=True, exist_ok=True)

    member_signals = _compiled_signals(book, panel)

    entries: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for i, spec in enumerate(specs):
        subset = [member_signals[j] for j in spec["subset"] if j < len(member_signals)]
        if not subset:
            log.warning("freeze spec %d has an empty subset — skipped", i)
            continue
        pred = _combined(subset, panel, is_mask, spec["model"], target_horizon)
        if pred is None:
            log.warning("freeze spec %d (%s) could not fit — skipped", i, spec["model"])
            continue
        fname = f"signal_{len(entries)}.parquet"
        pred.to_parquet(directory / fname)
        entries.append({
            "file": fname,
            "model": spec["model"],
            "subset": [book[j]["factor_id"] for j in spec["subset"] if j < len(book)],
        })

        if audit:
            audits.append(_poison_audit_one(subset, panel, split, spec["model"],
                                            target_horizon, pred))

    if not entries:
        raise ValueError("no evaluation signal could be frozen (all specs failed)")

    manifest = {
        "version": int(version),
        "k": len(entries),
        "target_horizon": int(target_horizon),
        "book_hash": _book_hash(book),
        "book_ids": [p.get("factor_id") for p in book],
        "panel_window_key": _window_key(close),
        "split_sizes": dict(split.sizes),
        "signals": entries,
        "poison_audit": {
            "audited": bool(audit),
            "passed": all(a["passed"] for a in audits) if audits else None,
            "per_signal": audits,
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # Deliberately do NOT cache the in-memory frames: load() always reads the
    # persisted parquet, so every consumer scores exactly the artifact on disk
    # (a parquet round-trip normalises index metadata like `freq`).
    return FrozenSignalSet(directory=directory, manifest=manifest)


def _poison_audit_one(subset: Sequence[pd.DataFrame], panel: dict[str, Any],
                      split: Any, model: str, target_horizon: int,
                      reference: pd.DataFrame) -> dict[str, Any]:
    """Poison every VAL row of the panel → IS-row predictions must not move.

    Proves the frozen signal's *fit* consumed no information beyond IS (the
    label-availability mask must have dropped every IS row whose ``t+h`` label
    would read a VAL price).  Feature values at VAL rows legitimately change
    the VAL-row *predictions* — those are causal per-row inputs — so only the
    IS rows are compared.
    """
    is_mask = np.asarray(split.is_mask, dtype=bool)
    val_mask = np.asarray(split.val_mask, dtype=bool)

    poisoned_panel = {}
    for k, df in panel.items():
        p = df.copy()
        p.loc[val_mask] = p.loc[val_mask] * 7.7 + 123.0  # deterministic corruption
        poisoned_panel[k] = p
    poisoned_subset = [s.copy() for s in subset]  # factor signals stay (audit isolates the fit)

    pred_poisoned = _combined(poisoned_subset, poisoned_panel, is_mask, model,
                              target_horizon)
    if pred_poisoned is None:
        return {"passed": False, "reason": "poisoned refit failed"}
    a = reference.to_numpy(dtype=float)[is_mask]
    b = pred_poisoned.to_numpy(dtype=float)[is_mask]
    both = np.isfinite(a) & np.isfinite(b)
    max_diff = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
    same_nan = bool(np.array_equal(np.isfinite(a), np.isfinite(b)))
    passed = bool(max_diff == 0.0 and same_nan)
    return {"passed": passed, "max_is_row_diff": max_diff, "model": model}
