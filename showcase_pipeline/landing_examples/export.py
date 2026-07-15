"""Write one example's full data pack into the company-brain marketing repo.

Per example ``<out_root>/<name>/``:
    card.json            – the verdict-card numbers + copy
    equity_curve.json    – raw IS/OOS series + pre-scaled SVG polyline points
    equity_curve.csv     – the raw stitched series
    card.png             – matplotlib fallback figure
    behind_the_verdict.md – the process teardown
    chat_transcript.json – grounded chat-demo sequence
    provenance.json      – config/run/git provenance

Also regenerates ``<out_root>/index.md``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .metrics import (
    CardMetrics,
    assemble_card_metrics,
    recompute_equity_curves,
    svg_points,
)
from .story import build_story_md
from .transcript import build_transcript, polish_with_llm
from .verdict import CAVEAT, assign_badge, check_compliance, verdict_note

log = logging.getLogger("landing_examples.export")

GOOD_COLOR, BAD_COLOR = "#45C08C", "#E8674C"
GROUND, SURFACE, TEXT = "#0A0E17", "#111A28", "#C9D6EA"


def _universe_label(universe: dict[str, Any]) -> str:
    span = [str(t)[:4] for t in universe.get("timespan", []) if t]
    bits = [universe.get("universe") or "custom universe",
            universe.get("frequency") or ""]
    if len(span) == 2:
        bits.append("–".join(span))
    return ", ".join(b for b in bits if b)


def build_card_json(candidate: dict[str, Any], card: CardMetrics,
                    badge: str, note: str) -> dict[str, Any]:
    universe = candidate.get("universe") or {}
    return {
        "name": card.strategy_name,
        "universe": _universe_label(universe),
        "badge": badge,
        "source": "delegate",  # the fund proposed it; "your idea" mode is a
                               # product surface, not something this run has
        "stats": {
            "pbo_pct": round(card.pbo * 100) if card.pbo is not None else None,
            "pbo_n_splits": card.pbo_n_splits,
            "is_sharpe": card.is_sharpe,
            "deflated_sharpe_prob": card.dsr_prob,
            "oos_sharpe": card.oos_sharpe,
            "oos_result": card.oos_result,
            "sharpe_decay_pct": card.sharpe_decay_pct,
            "n_trials": card.n_trials,
        },
        "note": note,
        "caveat": CAVEAT,
        "model_type": card.model_type,
        "n_factors": len(card.factor_ids),
        "target_horizon_bars": card.target_horizon,
    }


def render_card_png(curves: dict[str, Any], card: CardMetrics, badge: str,
                    out_path: Path) -> None:
    """Matplotlib fallback of the animated card chart (dark theme, IS|OOS)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    is_curve, oos_curve = curves["is_curve"], curves["oos_curve"]
    oos_color = GOOD_COLOR if badge == "Robust" else BAD_COLOR

    fig, ax = plt.subplots(figsize=(11, 4), dpi=130)
    fig.patch.set_facecolor(GROUND)
    ax.set_facecolor(SURFACE)
    ax.plot(is_curve.index, is_curve.values * 100, color=GOOD_COLOR, lw=2.0,
            label="In-sample")
    ax.plot(oos_curve.index, oos_curve.values * 100, color=oos_color, lw=2.0,
            label="Out-of-sample")
    ax.axvline(is_curve.index[-1], color="#96AACD", lw=1, ls="--", alpha=0.5)
    ax.text(oos_curve.index[0], ax.get_ylim()[1], "  OUT-OF-SAMPLE →",
            color="#96AACD", fontsize=8, va="top")
    ax.set_ylabel("Cumulative return (%)", color=TEXT)
    ax.set_title(f"{card.strategy_name} — {badge}", color=TEXT, fontsize=11)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2A3A52")
    ax.grid(alpha=0.12, color="#96AACD")
    ax.legend(facecolor=SURFACE, edgecolor="#2A3A52", labelcolor=TEXT, fontsize=8)
    fig.text(0.01, 0.01, CAVEAT, color="#96AACD", fontsize=6.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, bbox_inches="tight", facecolor=GROUND)
    plt.close(fig)


def _series_pairs(s: pd.Series) -> list[list[Any]]:
    return [[ts.isoformat(), round(float(v), 6)] for ts, v in s.items()]


def build_provenance(candidate: dict[str, Any], scope, curves: dict[str, Any],
                     transcript: dict[str, Any]) -> dict[str, Any]:
    from quant_fund_agent.config import get_settings, provenance_stamp

    try:
        stamp = provenance_stamp(scope.label, get_settings().data)
    except Exception as e:  # noqa: BLE001 — provenance must not block export
        log.warning("provenance_stamp failed (%s)", e)
        stamp = {"scope": scope.label}

    git_commit = None
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        git_commit = out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        pass

    evo_config = None
    evo_path = scope.dir / "evolution" / "run_config.json"
    if evo_path.exists():
        try:
            evo_config = json.loads(evo_path.read_text())
        except Exception:  # noqa: BLE001
            pass

    return {
        **stamp,
        "git_commit": git_commit,
        "prerun": scope.prerun,
        "attempt": candidate.get("attempt"),
        "strategy_id": candidate.get("strategy_id"),
        "approved_by_pipeline": candidate.get("approved"),
        "reject_stage": candidate.get("reject_stage"),
        "dumped_at": candidate.get("dumped_at"),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evolution_run_config": evo_config,
        "recompute_match": curves.get("recompute_match"),
        "oos_sharpe_recorded": curves.get("oos_sharpe_recorded"),
        "oos_sharpe_recomputed": curves.get("oos_sharpe_recomputed"),
        "llm_polished_transcript": transcript.get("llm_polished", False),
        "models": {
            "architect": os.getenv("ARCHITECT_LLM_MODEL", "gpt-4o-mini"),
            "statistician": os.getenv("STATISTICIAN_LLM_MODEL", "gpt-4o-mini"),
        },
    }


def export_example(
    candidate: dict[str, Any],
    name: str,
    out_root: Path,
    scope,
    curves: dict[str, Any] | None = None,
    polish_llm: bool = False,
) -> Path:
    """Write the full data pack for one candidate. Returns the example dir."""
    card = assemble_card_metrics(candidate)
    if not card.card_ready:
        raise ValueError(
            f"attempt {candidate.get('attempt')} is not card-ready "
            f"(rejected at {candidate.get('reject_stage')!r} before the "
            "statistical tests ran)")
    badge = assign_badge(card)
    note = verdict_note(card, badge)

    if curves is None:
        curves = recompute_equity_curves(candidate)
    if not curves.get("recompute_match", True):
        log.warning(
            "OOS recompute drift: recorded Sharpe %s vs recomputed %s — "
            "flagged in provenance",
            curves.get("oos_sharpe_recorded"), curves.get("oos_sharpe_recomputed"))

    transcript = build_transcript(candidate, card, badge, note)
    if polish_llm:
        transcript = polish_with_llm(transcript)
    story = build_story_md(candidate, card, badge, note, scope)

    out_dir = Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    card_json = build_card_json(candidate, card, badge, note)
    check_compliance(json.dumps(card_json))
    (out_dir / "card.json").write_text(json.dumps(card_json, indent=2))

    is_curve, oos_curve = curves["is_curve"], curves["oos_curve"]
    points = svg_points(is_curve, oos_curve)
    (out_dir / "equity_curve.json").write_text(json.dumps({
        "is": _series_pairs(is_curve),
        "oos": _series_pairs(oos_curve),
        "divider_ts": is_curve.index[-1].isoformat(),
        **points,
        "viewBox": "0 0 600 260",
        "note": "cumulative return; OOS re-based to continue the IS curve",
    }, indent=2))

    stitched = pd.concat([
        pd.DataFrame({"cum_return": is_curve, "segment": "is"}),
        pd.DataFrame({"cum_return": oos_curve, "segment": "oos"}),
    ])
    stitched.index.name = "timestamp"
    stitched.to_csv(out_dir / "equity_curve.csv")

    render_card_png(curves, card, badge, out_dir / "card.png")
    (out_dir / "behind_the_verdict.md").write_text(story)
    (out_dir / "chat_transcript.json").write_text(json.dumps(transcript, indent=2))

    provenance = build_provenance(candidate, scope, curves, transcript)
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str))

    log.info("exported %s → %s", name, out_dir)
    return out_dir


def rebuild_index(out_root: Path) -> Path:
    """Regenerate <out_root>/index.md from every example's card.json."""
    rows = []
    for card_path in sorted(Path(out_root).glob("*/card.json")):
        try:
            c = json.loads(card_path.read_text())
        except Exception:  # noqa: BLE001
            continue
        s = c.get("stats", {})
        pbo = f"{s['pbo_pct']}%" if s.get("pbo_pct") is not None else "n/a"
        dsr = (f"{s['deflated_sharpe_prob']:.2f}"
               if s.get("deflated_sharpe_prob") is not None else "n/a")
        rows.append(
            f"| [{c.get('name', card_path.parent.name)}]({card_path.parent.name}/) "
            f"| {c.get('badge', '?')} | {pbo} | {dsr} | {s.get('oos_result', '?')} "
            f"| {c.get('universe', '?')} |")

    lines = [
        "# Landing-page examples",
        "",
        "Real research runs, exported with full provenance by",
        "`QuantFundAgent/showcase_pipeline/landing_examples/`. Every number was",
        "produced by the deterministic evaluation harness — see each example's",
        "`provenance.json` and `behind_the_verdict.md`.",
        "",
        "| example | badge | PBO | deflated Sharpe (prob) | out-of-sample | universe |",
        "|---------|-------|-----|------------------------|---------------|----------|",
        *rows,
        "",
        f"*{CAVEAT}*",
        "",
    ]
    index = Path(out_root) / "index.md"
    index.write_text("\n".join(lines))
    return index
