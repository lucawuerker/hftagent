"""Build fixed factor books used as conditioning context for evolution.

The evolutionary loop normally scores a candidate against its own accepted
archive.  For exploratory runs it is useful to condition every candidate on a
pre-existing book without pretending those factors were discovered by the current
search.  This module builds that book from prior workspace factors using a
LassoCV feature-selection pass.
"""

from __future__ import annotations

import json
import logging
import ast
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("research_eval.prebook")


DEFAULT_CODE_FALLBACKS = (
    Path("data/backtests/sp100_inject/factor_code"),
    Path("quant_fund_agent/factors/researcher"),
)


def _resolve_code_path(record: dict[str, Any],
                       fallback_dirs: Sequence[Path] = DEFAULT_CODE_FALLBACKS) -> Path | None:
    fid = record.get("id") or record.get("factor_id")
    candidates: list[Path] = []
    raw = record.get("code_path")
    if raw:
        candidates.append(Path(raw))
    if fid:
        candidates.extend(Path(d) / f"{fid}.py" for d in fallback_dirs)
    for path in candidates:
        if path.exists():
            return path
    return None


def _record_horizon(record: dict[str, Any]) -> int:
    raw = record.get("prediction_horizon")
    if raw is None:
        raw = (record.get("metadata") or {}).get("ic_target_horizon")
    try:
        horizon = int(raw)
        return horizon if horizon > 0 else 6
    except (TypeError, ValueError):
        return 6


def _ensure_prediction_horizon(code: str, factor_id: str, horizon: int) -> str:
    """Inject ``prediction_horizon`` for old archived factors that predate it."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = {
            (b.attr if isinstance(b, ast.Attribute) else getattr(b, "id", ""))
            for b in node.bases
        }
        if "BaseFactor" not in base_names:
            continue
        has_id = False
        has_horizon = False
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            names = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            has_id = has_id or (
                "factor_id" in names
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value == factor_id
            )
            has_horizon = has_horizon or ("prediction_horizon" in names)
        if not has_id or has_horizon:
            continue

        lines = code.splitlines()
        insert_at = node.lineno
        if node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                insert_at = getattr(first, "end_lineno", first.lineno)
            indent_line = lines[first.lineno - 1] if first.lineno - 1 < len(lines) else ""
            indent = re.match(r"\s*", indent_line).group(0) or "    "
        else:
            indent = "    "
        lines.insert(insert_at, f"{indent}prediction_horizon = {int(horizon)}")
        return "\n".join(lines) + ("\n" if code.endswith("\n") else "")
    return code


def load_workspace_programs(
    workspace: str | Path,
    *,
    preruns: Sequence[str] | None = None,
    fallback_dirs: Sequence[str | Path] = DEFAULT_CODE_FALLBACKS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load factor source programs from every factor DB in ``workspace``.

    Returns ``(programs, dropped)``.  Each program is directly consumable by the
    research-eval book API: it has at least ``factor_id`` and ``code``.
    """
    workspace = Path(workspace)
    wanted = set(preruns or [])
    fallback_paths = [Path(p) for p in fallback_dirs]
    programs: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    seen: set[str] = set()

    for db_path in sorted((workspace / "preruns").glob("*/factors/factor_db.json")):
        prerun = db_path.parents[1].name
        if wanted and prerun not in wanted:
            continue
        try:
            records = json.loads(db_path.read_text()).get("factors", [])
        except Exception as e:  # noqa: BLE001
            dropped.append({"factor_id": None, "prerun": prerun,
                            "reason": f"factor DB unreadable: {e}"})
            continue
        for rec in records:
            fid = rec.get("id")
            if not fid or fid in seen:
                continue
            seen.add(fid)
            code_path = _resolve_code_path(rec, fallback_paths)
            if code_path is None:
                dropped.append({"factor_id": fid, "prerun": prerun,
                                "reason": "source code not found"})
                continue
            try:
                code = code_path.read_text()
            except Exception as e:  # noqa: BLE001
                dropped.append({"factor_id": fid, "prerun": prerun,
                                "reason": f"source code unreadable: {e}"})
                continue
            horizon = _record_horizon(rec)
            code = _ensure_prediction_horizon(code, fid, horizon)
            programs.append({
                "factor_id": fid,
                "code": code,
                "name": rec.get("name") or fid,
                "category": rec.get("category") or "other",
                "description": rec.get("description") or "",
                "trading_idea": rec.get("trading_idea") or "",
                "prediction_horizon": horizon,
                "source_prerun": prerun,
                "source_db": str(db_path),
                "code_path": str(code_path),
                "ic_at_target": (rec.get("metadata") or {}).get("ic_at_target"),
            })
    return programs, dropped


def _flatten_signal(program: dict[str, Any], panel: dict[str, pd.DataFrame]) -> np.ndarray:
    from quant_fund_agent.comparison.standardize import per_underlying_zscore
    from quant_fund_agent.factors.inmem import compile_factor, compute_signal

    cls = compile_factor(program["code"], program["factor_id"])
    available = set(panel)
    missing = [f for f in (getattr(cls, "inputs", None) or ["close"]) if f not in available]
    if missing:
        raise KeyError(f"panel lacks required field(s) {missing}")
    sig = compute_signal(cls, panel)
    close = panel["close"]
    z = per_underlying_zscore(
        sig.reindex(index=close.index, columns=close.columns).replace([np.inf, -np.inf], np.nan)
    )
    return z.to_numpy(dtype=float).ravel()


def fit_lasso_prebook(
    programs: Sequence[dict[str, Any]],
    panel: dict[str, pd.DataFrame],
    *,
    target_horizon: int = 6,
    max_rows: int = 50_000,
    max_members: int | None = 12,
    seed: int = 7,
    min_abs_coef: float = 1e-12,
) -> dict[str, Any]:
    """Fit LassoCV over ``programs`` and return a fixed-book JSON payload."""
    from quant_fund_agent.backtesting.data_loader import forward_returns

    close = panel["close"]
    y = forward_returns(close, horizon=target_horizon).to_numpy(dtype=float).ravel()
    columns: list[np.ndarray] = []
    usable: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for i, program in enumerate(programs, start=1):
        fid = program.get("factor_id", "?")
        if i == 1 or i % 25 == 0 or i == len(programs):
            log.info("prebook signals: %d/%d", i, len(programs))
        try:
            col = _flatten_signal(program, panel)
            if not np.isfinite(col).any():
                raise ValueError("all-NaN signal")
        except Exception as e:  # noqa: BLE001
            dropped.append({"factor_id": fid, "reason": str(e)[:300]})
            continue
        usable.append(dict(program))
        columns.append(col)

    if not usable:
        return {
            "ok": False,
            "error": "no workspace factor produced a usable signal",
            "n_candidates": len(programs),
            "dropped": dropped,
        }

    X = np.column_stack(columns)
    valid = np.isfinite(y)
    Xv = np.nan_to_num(X[valid], nan=0.0, posinf=0.0, neginf=0.0)
    yv = y[valid]
    if len(yv) > max_rows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(yv), size=max_rows, replace=False))
        Xv, yv = Xv[idx], yv[idx]
    if len(yv) < 50:
        return {
            "ok": False,
            "error": f"too few labelled rows for Lasso ({len(yv)})",
            "n_candidates": len(programs),
            "n_usable": len(usable),
            "dropped": dropped,
        }

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.preprocessing import StandardScaler
    import warnings

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    Xs = x_scaler.fit_transform(Xv)
    ys = y_scaler.fit_transform(yv.reshape(-1, 1)).ravel()
    alpha_grid = np.logspace(-5, -1, 80)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        est = LassoCV(alphas=alpha_grid, cv=4, max_iter=50_000,
                      random_state=seed, n_jobs=-1)
        est.fit(Xs, ys)
    coef = np.asarray(est.coef_, dtype=float).ravel()
    alpha = float(est.alpha_)
    alpha_source = "cv"
    if not np.any(np.abs(coef) > min_abs_coef):
        alpha = float(alpha_grid.min())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            fallback = Lasso(alpha=alpha, max_iter=100_000, random_state=seed)
            fallback.fit(Xs, ys)
        coef = np.asarray(fallback.coef_, dtype=float).ravel()
        alpha_source = "fallback_min_alpha"
    if coef.size != len(usable):
        return {
            "ok": False,
            "error": f"Lasso coefficient shape mismatch ({coef.size} != {len(usable)})",
            "n_candidates": len(programs),
            "n_usable": len(usable),
            "dropped": dropped,
        }

    abs_coef = np.abs(coef)
    selected_idx = np.flatnonzero(abs_coef > min_abs_coef)
    selected_idx = selected_idx[np.argsort(abs_coef[selected_idx])[::-1]]
    if max_members is not None:
        selected_idx = selected_idx[:max(0, int(max_members))]
    total = float(abs_coef[selected_idx].sum()) if len(selected_idx) else 0.0

    selected = []
    for rank, j in enumerate(selected_idx, start=1):
        program = dict(usable[int(j)])
        program["coefficient"] = float(coef[int(j)])
        program["abs_coefficient"] = float(abs_coef[int(j)])
        program["importance"] = float(abs_coef[int(j)] / total) if total > 0 else 0.0
        program["rank"] = rank
        selected.append(program)

    return {
        "ok": True,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "lasso",
        "target_horizon": int(target_horizon),
        "max_rows": int(max_rows),
        "max_members": max_members,
        "seed": int(seed),
        "alpha": alpha,
        "alpha_source": alpha_source,
        "alpha_grid": [float(x) for x in alpha_grid],
        "n_candidates": len(programs),
        "n_usable": len(usable),
        "n_labelled_rows_fit": int(len(yv)),
        "n_lasso_nonzero": int((abs_coef > min_abs_coef).sum()),
        "n_selected": len(selected),
        "selected_factor_ids": [p["factor_id"] for p in selected],
        "factors": selected,
        "dropped": dropped,
    }


def book_entries(prebook: dict[str, Any]) -> list[dict[str, str]]:
    """Return the evaluator's compact ``book`` shape from a prebook payload."""
    return [
        {"factor_id": f["factor_id"], "code": f["code"]}
        for f in prebook.get("factors", [])
        if f.get("factor_id") and f.get("code")
    ]


def save_prebook(prebook: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prebook, indent=2, default=str))
    return path


def load_prebook(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
