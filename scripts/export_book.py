#!/usr/bin/env python
"""Export a finished research prerun as a portable book bundle.

The product side (lagias/quant-engine ``scripts/import_book.py``) consumes the
bundle per ``docs/factor-book/BOOK_AND_MIGRATION.md`` §3: dedup against the
product book, re-validate on the product config, curation + publish deflation
with SUMMED n_trials.  Nothing is transformed at export time — provenance
travels whole.

Included per prerun: factors/ (factor_db.json + code), evolution/{state.json,
run_config.json, lineage.jsonl, llm_usage.json, prequential.jsonl,
gen_quality.jsonl}, manifest.json, papers_read.json, plus the scope's
config.snapshot.json and a bundle_meta.json (scope, git commit, n_trials).
Deliberately EXCLUDED (analysis artifacts, not research state): figures,
book_backtest*, persona_strategies*, prequential_deployment, *.log,
llm_transcript.jsonl.

    ./venv/bin/python scripts/export_book.py --prerun L4_terra_s0 \
        [--scope fmp_archive_equity_nasdaq100pit] [--out data/book_bundles]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PRERUN_FILES = [
    "manifest.json",
    "papers_read.json",
    "evolution/state.json",
    "evolution/run_config.json",
    "evolution/lineage.jsonl",
    "evolution/llm_usage.json",
    "evolution/prequential.jsonl",
    "evolution/gen_quality.jsonl",
    "evolution/progressive.json",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prerun", required=True)
    p.add_argument("--scope", default="fmp_archive_equity_nasdaq100pit")
    p.add_argument("--out", default="data/book_bundles")
    args = p.parse_args()

    scope_dir = REPO / "data" / "workspaces" / args.scope
    prerun_dir = scope_dir / "preruns" / args.prerun
    if not (prerun_dir / "factors" / "factor_db.json").exists():
        raise SystemExit(f"no factor_db.json under {prerun_dir}")

    manifest = json.loads((prerun_dir / "manifest.json").read_text()) \
        if (prerun_dir / "manifest.json").exists() else {}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = None
    meta = {
        "bundle_version": 1,
        "prerun": args.prerun,
        "scope": args.scope,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "source_repo": "QuantFundAgent (thesis)",
        "git_commit": commit,
        "n_trials": manifest.get("n_trials"),
        "n_factors": manifest.get("n_factors"),
        "llm_model": manifest.get("llm_model"),
        "engine": manifest.get("engine"),
    }

    # Self-contained member list: prerun factor records point at code files in
    # the shared factors/researcher package (the prerun's own factors/code dir
    # is empty), so the code must be inlined at export time or the bundle is
    # codeless on the product side.
    fdb = json.loads((prerun_dir / "factors" / "factor_db.json").read_text())
    members, missing = [], []
    for r in fdb.get("factors") or []:
        code_path = Path(r.get("code_path") or "")
        if not code_path.is_absolute():
            code_path = REPO / code_path
        if not code_path.exists():
            # records written on another machine carry that machine's absolute
            # path — re-resolve by basename inside this repo's shared package
            local = (REPO / "quant_fund_agent" / "factors" / "researcher"
                     / code_path.name)
            code_path = local
        if not code_path.exists():
            missing.append(r["id"])
            continue
        members.append({
            "factor_id": r["id"],
            "code": code_path.read_text(),
            "category": r.get("category") or "other",
            "prediction_horizon": r.get("prediction_horizon") or 6,
            "name": r.get("name") or r["id"],
            "trading_idea": r.get("trading_idea") or "",
            "source_prerun": args.prerun,
        })
    if missing:
        raise SystemExit(f"{len(missing)} factor code file(s) missing "
                         f"(e.g. {missing[:3]}) — refusing a codeless bundle")
    meta["n_members_with_code"] = len(members)

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / f"book_bundle_{args.prerun}.tar.gz"
    root = f"book_bundle_{args.prerun}"

    with tarfile.open(bundle, "w:gz") as tf:
        def add(path: Path, arcname: str) -> None:
            if path.exists():
                tf.add(path, arcname=f"{root}/{arcname}")

        add(scope_dir / "config.snapshot.json", "config.snapshot.json")
        add(prerun_dir / "factors", f"prerun/{args.prerun}/factors")
        for rel in PRERUN_FILES:
            add(prerun_dir / rel, f"prerun/{args.prerun}/{rel}")
        for fname, payload in (("bundle_meta.json", meta),
                               ("book_members.json", {"members": members})):
            tmp = out_dir / f".tmp_{args.prerun}_{fname}"
            tmp.write_text(json.dumps(payload, indent=2))
            tf.add(tmp, arcname=f"{root}/{fname}")
            tmp.unlink()

    size_mb = bundle.stat().st_size / 1e6
    print(f"wrote {bundle}  ({size_mb:.1f} MB)  "
          f"n_factors={meta['n_factors']} n_trials={meta['n_trials']}")


if __name__ == "__main__":
    main()
