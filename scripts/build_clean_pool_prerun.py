"""Build a pseudo-prerun containing an arm's KEPT-POOL factors with
level_rho < --max-rho (default 0.9), writing each factor's code to a file so
the oneshot-style loaders (load_book / PIT race availability=full) work.

Rho source: the harness diagnostic in state.json where present, else
data/comparisons/wf_book_analysis/derived/pool_level_profiles.csv.

Usage: --arm LDP_terra_s0 [--suffix CLN] [--max-rho 0.9]
Creates data/workspaces/.../preruns/<arm-prefix><suffix>_terra_s0/factors/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WS = REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"
PROFILES = REPO / "data/comparisons/wf_book_analysis/derived/pool_level_profiles.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--suffix", default="CLN")
    ap.add_argument("--max-rho", type=float, default=0.9)
    args = ap.parse_args()

    st = json.load((WS / args.arm / "evolution/state.json").open())
    rho_by_fid: dict[str, float] = {}
    entries = {}
    for e in st.get("kept_pool", []) + st.get("archive", []):
        prog = e["genome"]["programs"][0]
        fid = prog["factor_id"]
        entries.setdefault(fid, prog)
        r = (e["fitness"].get("diagnostics") or {}).get("level_rho")
        if r is not None:
            rho_by_fid.setdefault(fid, float(r))
    if not rho_by_fid and PROFILES.exists():
        import csv
        for row in csv.DictReader(PROFILES.open()):
            if row["arm"] == args.arm:
                rho_by_fid[row["factor_id"]] = float(row["rho_med"])

    keep = [fid for fid in entries
            if fid in rho_by_fid and rho_by_fid[fid] < args.max_rho]
    base = args.arm.split("_terra")[0].split("_4omini")[0]
    tail = "_terra_s0" if "_terra" in args.arm else "_4omini_s0"
    name = f"{base}{args.suffix}{tail}"
    tgt = WS / name
    code_dir = tgt / "factors/code"
    code_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for fid in keep:
        path = code_dir / f"{fid}.py"
        path.write_text(entries[fid]["code"])
        records.append({"id": fid, "code_path": str(path),
                        "inputs": [], "metadata": {"pool_clean": True}})
    (tgt / "factors/factor_db.json").write_text(
        json.dumps({"factors": records}))
    print(f"{args.arm}: pool {len(entries)}, rho known {len(rho_by_fid)}, "
          f"kept {len(keep)} (rho<{args.max_rho}) -> {name}")


if __name__ == "__main__":
    main()
