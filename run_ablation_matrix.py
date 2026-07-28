"""Ablation-matrix orchestrator for the final thesis comparison run.

Drives the run matrix from docs/research-evolution/FINAL_RUN_PLAN.md: the
ablation ladder (L0-L7) on the workhorse model plus the model sweep at the full
configuration, each run in its own subprocess (fresh memory, per-run env), with
resumability, per-run cost ceilings, and a preflight that refuses to spend
credits on a broken setup.

Usage
-----
    ./venv/bin/python run_ablation_matrix.py --plan matrix/final_matrix.yaml --preflight-only
    ./venv/bin/python run_ablation_matrix.py --plan matrix/final_matrix.yaml
    ./venv/bin/python run_ablation_matrix.py --plan matrix/final_matrix.yaml --only L5_opus5_s0,L5_sol_s0

The plan file is YAML::

    config_file: quant.config.nasdaq100_2010.yaml
    defaults:                 # merged into every evolution arm's flags
      generations: 20
      mechanism-groups: 4
      demes-per-group: 3
      children-per-deme: 4
      population: 16
      progressive-reveal: true
      reveal-every: 2
      test-frac: 0.2
      curation: greedy
      selection-deflation: "on"
      archive-cap: 25
      n-tickers: 0
      max-cost-usd: 120
    providers:                # env applied per model key
      opus5:  {model: "anthropic.claude-opus-5",  llm-provider: bedrock_converse}
      sol:    {model: "gpt-5.6-sol"}
      luna:   {model: "gpt-5.6-luna"}
      sonnet5: {model: "anthropic.claude-sonnet-5", llm-provider: bedrock_converse}
      fable5: {model: "anthropic.claude-fable-5", llm-provider: bedrock_converse}
      muse:   {model: "muse-spark-1.1",
               env: {FACTOR_RESEARCH_LLM_BASE_URL: "https://api.llama.meta.com/v1",
                     OPENAI_API_KEY: "${META_API_KEY}"}}
    arms:
      - {name: L2_opus5_s0, provider: opus5, seed: 0,
         flags: {retrieval: none, mechanism-groups: 1}}
      - {name: L5_opus5_s0, provider: opus5, seed: 0,
         flags: {retrieval: graphrag, debate: "on"}}
      # entrypoint: gp | oneshot | evolution (default evolution)
      - {name: L0_gp_s0, entrypoint: gp, seed: 0, flags: {}}

Every run writes into its own prerun (``--name <arm name>``); a run whose
``status.json`` says ``ok`` is skipped on re-invocation, so the sweep is
resumable after a crash. Each subprocess gets ``QF_USE_MCP=0`` (in-process
scoring — the ~20s panel load is paid once per run, and a dead MCP child can't
strand the panel), a per-run LLM transcript, and the plan's cost ceiling.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
STATUS_OK = "ok"


# ---------------------------------------------------------------------------
# plan handling
# ---------------------------------------------------------------------------

def load_plan(path: str) -> dict:
    plan = yaml.safe_load(Path(path).read_text())
    for key in ("config_file", "providers", "arms"):
        if key not in plan:
            raise SystemExit(f"plan file is missing required key: {key}")
    return plan


def _expand_env(value: str) -> str:
    """Expand ``${VAR}`` references from the orchestrator's environment."""
    return os.path.expandvars(value)


def arm_command(plan: dict, arm: dict) -> tuple[list[str], dict, str]:
    """Return (argv, extra_env, prerun_name) for one arm."""
    name = arm["name"]
    entry = arm.get("entrypoint", "evolution")
    seed = arm.get("seed", 0)
    env: dict[str, str] = {}

    flags = copy.deepcopy(plan.get("defaults", {}))
    flags.update(arm.get("flags", {}))

    provider_key = arm.get("provider")
    model = None
    if provider_key is not None:
        prov = plan["providers"][provider_key]
        model = prov.get("model")
        if prov.get("llm-provider"):
            flags["llm-provider"] = prov["llm-provider"]
        for k, v in (prov.get("env") or {}).items():
            env[k] = _expand_env(str(v))

    if entry == "evolution":
        argv = [sys.executable, "run_factor_evolution.py", "--name", name,
                "--config", plan["config_file"], "--seed", str(seed)]
        if model:
            argv += ["--model", model]
        for key, val in flags.items():
            flag = f"--{key}"
            if isinstance(val, bool):
                if val:
                    argv.append(flag)
            else:
                argv += [flag, str(val)]
    elif entry == "gp":
        argv = [sys.executable, "run_gp_factor_mining.py", "--name", name,
                "--config-file", plan["config_file"], "--seed", str(seed)]
        for key, val in flags.items():
            argv += [f"--{key}", str(val)] if not isinstance(val, bool) else ([f"--{key}"] if val else [])
    elif entry == "oneshot":
        argv = [sys.executable, "run_factor_research.py", "--name", name]
        if model:
            argv += ["--model", model]
        env["QF_CONFIG_FILE"] = plan["config_file"]
        for key, val in flags.items():
            argv += [f"--{key}", str(val)] if not isinstance(val, bool) else ([f"--{key}"] if val else [])
    else:
        raise SystemExit(f"unknown entrypoint {entry!r} for arm {name}")
    return argv, env, name


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def preflight(plan: dict, run_probes: bool = True) -> None:
    """Refuse to spend credits on a broken setup."""
    os.environ["QF_CONFIG_FILE"] = plan["config_file"]
    from quant_fund_agent.config import get_settings
    from quant_fund_agent.data.panel import load_panel

    settings = get_settings()
    print(f"[preflight] config={plan['config_file']} provider={settings.data.provider} "
          f"membership={settings.data.membership} {settings.data.start}->{settings.data.end}")

    # 1. forward reserve: panel must end >= 2 years before today
    end = dt.date.fromisoformat(str(settings.data.end))
    if end > dt.date.today() - dt.timedelta(days=2 * 365 - 5):
        raise SystemExit(f"[preflight] FORWARD RESERVE VIOLATED: config end {end} is within 2y of today")

    # 2. membership mask density
    panel = load_panel(settings.data, fields=["close"])
    close = panel["close"]
    density = float(close.notna().mean().mean())
    print(f"[preflight] panel {close.shape[0]} bars x {close.shape[1]} tickers, density {density:.3f}")
    if not (0.25 < density < 0.85):
        raise SystemExit("[preflight] membership mask density out of band — mask may have failed open")

    # 3. knowledge graph must exist for graphrag arms
    needs_graph = any((a.get("flags", {}).get("retrieval",
                       plan.get("defaults", {}).get("retrieval", "graphrag"))) == "graphrag"
                      and a.get("entrypoint", "evolution") == "evolution"
                      for a in plan["arms"])
    graph_path = REPO / "data/knowledge/graph.json"
    if needs_graph and not graph_path.exists():
        raise SystemExit("[preflight] graphrag arms present but data/knowledge/graph.json missing — "
                         "run scripts/build_knowledge_graph.py first")

    # 4. one-call provider probes through the real factory + meter
    if run_probes:
        from quant_fund_agent.llm import make_chat_llm, reset_usage, usage_summary
        for key, prov in plan["providers"].items():
            model = prov.get("model")
            saved = {k: os.environ.get(k) for k in (prov.get("env") or {})}
            try:
                for k, v in (prov.get("env") or {}).items():
                    os.environ[k] = _expand_env(str(v))
                reset_usage()
                llm = make_chat_llm(model=model, provider=prov.get("llm-provider"),
                                    temperature=0.0, timeout=60, max_retries=1)
                out = llm.invoke("Reply with the single word OK.")
                usage = usage_summary()["total"]
                assert usage["input_tokens"] > 0, "meter recorded no tokens"
                print(f"[preflight] probe {key}:{model} -> {getattr(out, 'content', out)!r:.40} "
                      f"({usage['input_tokens']}in/{usage['output_tokens']}out, ${usage['cost_usd']:.5f})")
            except Exception as exc:  # noqa: BLE001 - report and abort
                raise SystemExit(f"[preflight] provider probe FAILED for {key} ({model}): {exc}") from exc
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
    print("[preflight] all checks passed")


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

def scope_dir(plan: dict, prerun: str) -> Path:
    os.environ["QF_CONFIG_FILE"] = plan["config_file"]
    from quant_fund_agent.config import get_settings, default_config_name
    cfg_name = default_config_name(get_settings().data)
    return REPO / "data" / "workspaces" / cfg_name / "preruns" / prerun


def run_arm(plan: dict, arm: dict, log_dir: Path) -> dict:
    argv, extra_env, name = arm_command(plan, arm)
    sdir = scope_dir(plan, name)
    status_path = sdir / "orchestrator_status.json"
    if status_path.exists() and json.loads(status_path.read_text()).get("status") == STATUS_OK:
        print(f"[skip] {name} already ok")
        return {"name": name, "status": "skipped"}

    env = os.environ.copy()
    env.update(extra_env)
    env.setdefault("QF_USE_MCP", "0")
    env["QF_LLM_TRANSCRIPT_PATH"] = str(sdir / "evolution" / "llm_transcript.jsonl")
    log_path = log_dir / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    (sdir / "evolution").mkdir(parents=True, exist_ok=True)

    print(f"[run] {name}: {' '.join(argv)}")
    t0 = time.time()
    with open(log_path, "w") as log_fh:
        proc = subprocess.run(argv, cwd=REPO, env=env, stdout=log_fh,
                              stderr=subprocess.STDOUT, check=False)
    elapsed = time.time() - t0
    status = {"status": STATUS_OK if proc.returncode == 0 else "failed",
              "returncode": proc.returncode, "elapsed_sec": round(elapsed, 1),
              "argv": argv, "finished_at": dt.datetime.now().isoformat(timespec="seconds")}
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2))
    print(f"[done] {name}: {status['status']} in {elapsed/60:.1f} min (log: {log_path})")
    return {"name": name, **status}


def aggregate(plan: dict, results: list[dict]) -> None:
    rows = []
    for arm in plan["arms"]:
        sdir = scope_dir(plan, arm["name"])
        manifest = sdir / "manifest.json"
        row = {"arm": arm["name"], "provider": arm.get("provider"),
               "seed": arm.get("seed", 0)}
        if manifest.exists():
            m = json.loads(manifest.read_text())
            usage = (m.get("llm_usage") or {}).get("total", {})
            row.update(n_factors=m.get("n_factors"), n_trials=m.get("n_trials"),
                       llm_model=m.get("llm_model"),
                       cost_usd=round(usage.get("cost_usd", 0.0), 2),
                       tokens_in=usage.get("input_tokens"),
                       tokens_out=usage.get("output_tokens"))
        st = sdir / "orchestrator_status.json"
        if st.exists():
            s = json.loads(st.read_text())
            row.update(status=s.get("status"), elapsed_min=round((s.get("elapsed_sec") or 0) / 60, 1))
        rows.append(row)
    out = REPO / "data" / "comparisons" / "final_matrix"
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out / "matrix_summary.csv", index=False)
    print(df.to_string(index=False))
    print(f"[aggregate] wrote {out / 'matrix_summary.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", required=True, help="YAML plan file (see module docstring)")
    ap.add_argument("--only", default=None, help="comma list of arm names to run")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--no-probes", action="store_true",
                    help="skip the 1-call provider probes in preflight")
    ap.add_argument("--log-dir", default="data/comparisons/final_matrix/logs")
    args = ap.parse_args()

    plan = load_plan(args.plan)
    preflight(plan, run_probes=not args.no_probes)
    if args.preflight_only:
        return

    only = set(args.only.split(",")) if args.only else None
    results = []
    for arm in plan["arms"]:
        if only and arm["name"] not in only:
            continue
        results.append(run_arm(plan, arm, Path(args.log_dir)))
    aggregate(plan, results)


if __name__ == "__main__":
    main()
