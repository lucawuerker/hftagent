"""Render the "Behind the verdict" markdown for one exported example.

The teardown shows the full honest process: the plain-English trading ideas
(the LLM hypotheses stored on the researched factors), the generated factor
code, the evolution gate verdicts, the Architect's trial table, and the
explicit deflation / PBO / OOS arithmetic behind the badge.  Doubles as an
"Is it overfit?" social post and a methods/transparency page.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .metrics import CardMetrics
from .verdict import CAVEAT, check_compliance

log = logging.getLogger("landing_examples.story")


# ---------------------------------------------------------------------------
# Factor + evolution provenance lookups (all best-effort: a missing artifact
# degrades the story, never crashes the export)
# ---------------------------------------------------------------------------

def _load_factor_records(scope) -> dict[str, dict[str, Any]]:
    """Factor records by id from the scope's composed DB (seeds + researcher)."""
    records: dict[str, dict[str, Any]] = {}
    for path in (scope.composed_db_path, scope.factor_db_path):
        if not Path(path).exists():
            continue
        try:
            db = json.loads(Path(path).read_text())
        except Exception as e:  # noqa: BLE001
            log.warning("unreadable factor DB %s (%s)", path, e)
            continue
        for f in db.get("factors", []):
            records.setdefault(f.get("id"), f)
    return records


def _factor_code(record: dict[str, Any], scope) -> str | None:
    """The factor's Python source: code_path first, then the scope snapshot."""
    fid = record.get("id", "")
    candidates = []
    if record.get("code_path"):
        candidates.append(Path(record["code_path"]))
    candidates.append(scope.factor_code_dir / f"{fid}.py")
    for p in candidates:
        try:
            if p.exists():
                return p.read_text()
        except Exception:  # noqa: BLE001
            continue
    return None


def _evolution_lineage(scope, factor_ids: list[str]) -> dict[str, dict[str, Any]]:
    """lineage.jsonl rows keyed by factor id (evolution preruns only)."""
    lineage_path = scope.dir / "evolution" / "lineage.jsonl"
    if not lineage_path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    wanted = set(factor_ids)
    try:
        for line in lineage_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for fid in row.get("factor_ids", []):
                if fid in wanted and fid not in out:
                    out[fid] = row
    except Exception as e:  # noqa: BLE001
        log.warning("unreadable lineage %s (%s)", lineage_path, e)
    return out


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _fmt(x: Any, nd: int = 2) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "–"


def _trial_table(trial_history: list[dict[str, Any]]) -> str:
    lines = ["| trial | model | features | IS Sharpe |",
             "|------:|-------|---------:|----------:|"]
    for t in trial_history:
        spec = t.get("spec") or {}
        m = t.get("metrics") or {}
        n_feat = len(spec.get("factor_ids") or spec.get("weights") or [])
        lines.append(f"| {t.get('iteration', '?')} | {spec.get('model_type', '?')} "
                     f"| {n_feat} | {_fmt(m.get('sharpe_ratio'))} |")
    return "\n".join(lines)


def build_story_md(
    candidate: dict[str, Any],
    card: CardMetrics,
    badge: str,
    note: str,
    scope,
    max_factor_code: int = 3,
) -> str:
    """The full teardown markdown. Every sentence is deterministic."""
    universe = candidate.get("universe") or {}
    factor_ids = card.factor_ids
    records = _load_factor_records(scope)
    lineage = _evolution_lineage(scope, factor_ids)

    parts: list[str] = []
    parts.append(f"# Behind the verdict: {card.strategy_name}")
    parts.append("")
    parts.append(f"**Verdict: {badge}.** {note}")
    parts.append("")
    parts.append(
        f"This is a real research run — every number below was produced by a "
        f"deterministic evaluation harness, not by the AI that proposed the "
        f"strategy. Universe: {universe.get('universe', 'n/a')} "
        f"({universe.get('provider', 'n/a')}, {universe.get('frequency', 'n/a')}, "
        f"{'–'.join(str(t) for t in universe.get('timespan', []) if t)})."
    )
    parts.append("")

    # 1. The idea
    parts.append("## 1 · The idea")
    parts.append("")
    hyp = (candidate.get("hypothesis") or "").strip()
    if hyp:
        parts.append(f"> {hyp}")
        parts.append("")
    rationale = (candidate.get("selection_rationale") or "").strip()
    if rationale:
        parts.append(rationale)
        parts.append("")

    # 2. The researched factors (idea → code)
    parts.append("## 2 · The factors under the hood")
    parts.append("")
    parts.append(f"The strategy combines {len(factor_ids)} factor(s). Researched "
                 "factors were invented by an LLM, then compiled and scored by "
                 "the harness — the AI never grades its own work.")
    parts.append("")
    shown_code = 0
    for fid in factor_ids:
        rec = records.get(fid)
        if rec is None:
            parts.append(f"### `{fid}`")
            parts.append("")
            parts.append("_(seed/library factor — code in the main library)_")
            parts.append("")
            continue
        parts.append(f"### {rec.get('name') or fid} (`{fid}`)")
        parts.append("")
        idea = (rec.get("trading_idea") or rec.get("description") or "").strip()
        if idea:
            parts.append(f"*Trading idea:* {idea}")
            parts.append("")
        row = lineage.get(fid)
        if row:
            gates = row.get("gates") or {}
            obj = row.get("objective") or {}
            parts.append(
                f"*Evolution provenance:* generation {row.get('generation', '?')}, "
                f"operator `{row.get('operator', '?')}`; harness gates "
                f"{'passed' if gates.get('passed') else 'failed'} "
                f"(marginal value {_fmt(obj.get('marginal_value'), 4)}, "
                f"robustness {_fmt(obj.get('robustness'), 4)})."
            )
            parts.append("")
        if shown_code < max_factor_code:
            code = _factor_code(rec, scope)
            if code:
                parts.append("```python")
                parts.append(code.rstrip())
                parts.append("```")
                parts.append("")
                shown_code += 1

    # 3. The search the Architect ran
    trials = candidate.get("trial_history") or []
    parts.append("## 3 · The variants that were tried")
    parts.append("")
    parts.append(
        f"The strategy designer iterated {len(trials)} time(s) before settling on "
        "the final configuration. Every variant counts against the strategy in "
        "the deflation arithmetic below — trying more things makes a lucky "
        "backtest more likely, so the bar rises with every attempt."
    )
    parts.append("")
    if trials:
        parts.append(_trial_table(trials))
        parts.append("")

    # 4. The honest arithmetic
    parts.append("## 4 · The honest arithmetic")
    parts.append("")
    parts.append(f"- **In-sample Sharpe (final variant):** {_fmt(card.is_sharpe)}")
    if card.expected_max_sharpe is not None:
        parts.append(
            f"- **Expected best Sharpe by pure luck across {card.n_trials} "
            f"variants:** {_fmt(card.expected_max_sharpe)} — anything below "
            "this is indistinguishable from selection noise.")
    parts.append(
        f"- **Deflated Sharpe (probability the true edge is > 0 after "
        f"correcting for the {card.n_trials} variants tried and non-normal "
        f"returns):** {_fmt(card.dsr_prob)}")
    if card.pbo is not None:
        parts.append(
            f"- **Probability of backtest overfitting (CSCV over the "
            f"{card.n_trials} tried variants, {card.pbo_n_splits} splits):** "
            f"{card.pbo:.0%} — the chance that picking the best backtest picked "
            "a variant that underperforms out-of-sample.")
    else:
        parts.append("- **Probability of backtest overfitting:** not computable "
                     "for this run (needs at least two scored variants).")
    parts.append(
        f"- **Held-out test:** out-of-sample Sharpe {_fmt(card.oos_sharpe)} vs "
        f"{_fmt(card.oos_is_sharpe)} in-sample — the edge {card.oos_result}."
        + (f" (Sharpe decay {card.sharpe_decay_pct:.0f}%.)"
           if card.sharpe_decay_pct is not None else ""))
    parts.append("")

    # 5. Verdict
    parts.append("## 5 · The verdict")
    parts.append("")
    parts.append(f"**{badge}.** {note}")
    parts.append("")
    parts.append(
        "The badge is assigned by fixed thresholds on the harness numbers "
        "above; the AI that proposed the strategy has no influence on it.")
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append(f"*{CAVEAT}*")
    parts.append("")

    md = "\n".join(parts)
    return check_compliance(md)
