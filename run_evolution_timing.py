"""Granular timing / API-cost diagnostic for an evolutionary factor-research run.

Runs one *small* :class:`EvolutionLoop` with ``gpt-4o-mini`` and instruments it
end-to-end so you can see **where the wall-clock goes** and **what the API costs**
before committing to long runs.  Pure instrumentation: it monkeypatches functions
in-process and never edits the ``quant_fund_agent`` package.

Why this exists
---------------
A prior GP run (``run_gp_factor_mining.py --generations 6 --depth-schedule 3,5,7
--seed-pop 60``) took **>24 h**.  The evolutionary and GP arms share the
deterministic evaluation harness byte-for-byte, and code review points at the
marginal-value model-refit loop: per candidate the harness fits
``_combined_prediction`` for the with/without book of the marginal (LOCO) axis
and again for the jitter/perturbation probes, over a flattened
``(dev_bars × n_tickers)`` matrix whose feature count grows with the archive.
This script *measures* that.

It forces the evaluation to run **in-process** (``QF_USE_MCP=0``) so the
monkeypatches actually time the work, tags every LLM ``invoke`` for token/cost
accounting, and prints + writes (JSON + Markdown) a per-part breakdown with true
*self-time* (profiler-style, so the numbers are additive despite nesting).

Examples
--------
::

    # quick default (~minutes, a few cents)
    ./venv/bin/python run_evolution_timing.py

    # tiny smoke
    ./venv/bin/python run_evolution_timing.py --n-tickers 12 --generations 2 \
        --children-per-gen 4 --seed-ideas 4

    # linear combiner — should show _combined_prediction time collapse
    ./venv/bin/python run_evolution_timing.py --marginal-model ridge
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ── gpt-4o-mini pricing (USD per token).  Approximate, provider-published rates
#    as of Jan 2026 — adjust if OpenAI changes them. ──
GPT_4O_MINI_INPUT_USD_PER_TOKEN = 0.15 / 1_000_000
GPT_4O_MINI_OUTPUT_USD_PER_TOKEN = 0.60 / 1_000_000


# ════════════════════════════════════════════════════════════════════════════
# Instrumentation registry (true self-time via a call stack)
# ════════════════════════════════════════════════════════════════════════════

class Timer:
    """Records per-label ``count`` / ``inclusive`` / ``self`` seconds.

    A thread-local frame stack subtracts nested wrapped-call time from each
    frame, so ``self`` time is additive and (per call tree) sums to the root's
    inclusive time — exactly what makes "which part took how long" unambiguous
    even though ``evaluate_candidate`` → ``_marginal_value`` → ``_combined_prediction``
    are nested.
    """

    def __init__(self) -> None:
        self.stats: dict[str, dict[str, float]] = {}
        self._local = threading.local()

    def _stack(self) -> list[list[float]]:
        st = getattr(self._local, "stack", None)
        if st is None:
            st = self._local.stack = []
        return st

    @contextlib.contextmanager
    def span(self, label: str):
        stack = self._stack()
        frame = [0.0]  # accumulated child self-subtraction
        stack.append(frame)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            inclusive = time.perf_counter() - t0
            stack.pop()
            self_time = inclusive - frame[0]
            if stack:
                stack[-1][0] += inclusive  # this whole call is a child of the parent
            rec = self.stats.setdefault(
                label, {"count": 0.0, "inclusive": 0.0, "self": 0.0})
            rec["count"] += 1
            rec["inclusive"] += inclusive
            rec["self"] += self_time

    def wrap(self, label: str, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self.span(label):
                return fn(*args, **kwargs)
        return wrapper


TIMER = Timer()


# ── LLM token / cost / latency accounting ──
class LLMMeter:
    def __init__(self, max_cost_usd: float) -> None:
        self.max_cost_usd = max_cost_usd
        # role -> {calls, seconds, input_tokens, output_tokens, cost}
        self.by_role: dict[str, dict[str, float]] = {}
        self._role = threading.local()

    def role(self) -> str:
        return getattr(self._role, "value", "llm")

    @contextlib.contextmanager
    def as_role(self, role: str):
        prev = getattr(self._role, "value", None)
        self._role.value = role
        try:
            yield
        finally:
            self._role.value = prev if prev is not None else "llm"

    def record(self, seconds: float, resp: Any) -> None:
        in_tok, out_tok = _extract_tokens(resp)
        cost = (in_tok * GPT_4O_MINI_INPUT_USD_PER_TOKEN
                + out_tok * GPT_4O_MINI_OUTPUT_USD_PER_TOKEN)
        rec = self.by_role.setdefault(
            self.role(),
            {"calls": 0.0, "seconds": 0.0, "input_tokens": 0.0,
             "output_tokens": 0.0, "cost": 0.0})
        rec["calls"] += 1
        rec["seconds"] += seconds
        rec["input_tokens"] += in_tok
        rec["output_tokens"] += out_tok
        rec["cost"] += cost
        if self.total_cost() > self.max_cost_usd:
            raise RuntimeError(
                f"COST GUARD tripped: cumulative gpt-4o-mini cost "
                f"${self.total_cost():.4f} exceeded --max-cost-usd "
                f"${self.max_cost_usd:.2f}. Aborting the diagnostic run.")

    def total_cost(self) -> float:
        return sum(r["cost"] for r in self.by_role.values())

    def total_calls(self) -> int:
        return int(sum(r["calls"] for r in self.by_role.values()))


def _extract_tokens(resp: Any) -> tuple[int, int]:
    """Best-effort (input, output) token counts from a LangChain response."""
    usage = getattr(resp, "usage_metadata", None)
    if isinstance(usage, dict) and usage:
        return (int(usage.get("input_tokens", 0) or 0),
                int(usage.get("output_tokens", 0) or 0))
    meta = getattr(resp, "response_metadata", None) or {}
    tu = meta.get("token_usage") or meta.get("usage") or {}
    if tu:
        return (int(tu.get("prompt_tokens", tu.get("input_tokens", 0)) or 0),
                int(tu.get("completion_tokens", tu.get("output_tokens", 0)) or 0))
    return (0, 0)


# per-candidate evaluation samples: (book_size, seconds)
EVAL_SAMPLES: list[tuple[int, float]] = []


# ════════════════════════════════════════════════════════════════════════════
# Monkeypatching
# ════════════════════════════════════════════════════════════════════════════

def install_instrumentation(meter: LLMMeter) -> None:
    """Rebind module-global functions with timing wrappers (done once, up front)."""
    from quant_fund_agent.mcp import research_service as svc
    from quant_fund_agent.research_eval import harness
    from quant_fund_agent.agents.factor_research.evolution import loop as evo_loop

    # ── panel / signal compute (the cache-once + per-candidate signal cost) ──
    svc._load_panel_cached = TIMER.wrap("data.load_panel", svc._load_panel_cached)
    svc._cached_signal = TIMER.wrap("data.compute_signal", svc._cached_signal)

    # ── every evaluation metric / diagnostic in the harness ──
    #    _combined_prediction is the model-fit hotspot; its call count == #model fits.
    harness_fns = [
        "evaluate_candidate", "evaluate_set",
        "_combined_prediction", "_pooled_ic",
        "_marginal_value", "_residual_ic", "_independence",
        "_marginal_penalties", "_apply_marginal_penalties",
        "_structural_novelty", "_zoo_dedup", "_turnover_netcost",
        "_coverage",
    ]
    for name in harness_fns:
        fn = getattr(harness, name, None)
        if fn is not None:
            setattr(harness, name, TIMER.wrap(f"eval.{name}", fn))

    # ── LLM calls: patch BaseChatModel.invoke per concrete class, once, via the
    #    make_chat_llm factory (captures brainstorm, codegen, mutation, crossover
    #    uniformly, whatever code path builds the model). ──
    import quant_fund_agent.llm as llm_mod

    patched_classes: set[type] = set()
    orig_make = llm_mod.make_chat_llm

    def _patch_invoke(cls: type) -> None:
        if cls in patched_classes:
            return
        orig_invoke = cls.invoke

        @functools.wraps(orig_invoke)
        def timed_invoke(self, *a, **k):
            t0 = time.perf_counter()
            resp = orig_invoke(self, *a, **k)
            meter.record(time.perf_counter() - t0, resp)
            return resp

        cls.invoke = timed_invoke
        patched_classes.add(cls)

    @functools.wraps(orig_make)
    def make_chat_llm(*a, **k):
        model = orig_make(*a, **k)
        try:
            _patch_invoke(type(model))
        except Exception as e:  # noqa: BLE001 — never let instrumentation break the run
            logging.getLogger("timing").warning("could not patch invoke: %s", e)
        return model

    llm_mod.make_chat_llm = make_chat_llm

    # ── role tagging: wrap the loop functions that own an LLM code path so each
    #    invoke is attributed to brainstorm / codegen / mutate / crossover. ──
    orig_seed = evo_loop.seed_programs
    orig_codegen = evo_loop._codegen_program
    orig_semantic = evo_loop.EvolutionLoop._child_llm_semantic
    orig_crossover = evo_loop.EvolutionLoop._child_crossover
    orig_eval = evo_loop.EvolutionLoop.evaluate_program

    @functools.wraps(orig_seed)
    def seed_programs(*a, **k):
        with meter.as_role("brainstorm"), TIMER.span("phase.seed"):
            return orig_seed(*a, **k)

    @functools.wraps(orig_codegen)
    def _codegen_program(*a, **k):
        with meter.as_role("codegen"):
            return orig_codegen(*a, **k)

    @functools.wraps(orig_semantic)
    def _child_llm_semantic(self, *a, **k):
        with meter.as_role("mutate"):
            return orig_semantic(self, *a, **k)

    @functools.wraps(orig_crossover)
    def _child_crossover(self, *a, **k):
        with meter.as_role("crossover"):
            return orig_crossover(self, *a, **k)

    @functools.wraps(orig_eval)
    def evaluate_program(self, program):
        try:
            book_size = len(list(self.controller.archive_programs()))
        except Exception:  # noqa: BLE001
            book_size = -1
        t0 = time.perf_counter()
        with TIMER.span("phase.evaluate_candidate"):
            res = orig_eval(self, program)
        EVAL_SAMPLES.append((book_size, time.perf_counter() - t0))
        return res

    evo_loop.seed_programs = seed_programs
    evo_loop._codegen_program = _codegen_program
    evo_loop.EvolutionLoop._child_llm_semantic = _child_llm_semantic
    evo_loop.EvolutionLoop._child_crossover = _child_crossover
    evo_loop.EvolutionLoop.evaluate_program = evaluate_program


# ════════════════════════════════════════════════════════════════════════════
# Reporting
# ════════════════════════════════════════════════════════════════════════════

def _fmt(sec: float) -> str:
    if sec >= 60:
        return f"{sec/60:.1f}m"
    if sec >= 1:
        return f"{sec:.2f}s"
    return f"{sec*1000:.0f}ms"


def build_report(meter: LLMMeter, wall: float, summary: dict[str, Any],
                 args: argparse.Namespace) -> dict[str, Any]:
    stats = TIMER.stats
    eval_incl = stats.get("phase.evaluate_candidate", {}).get("inclusive", 0.0)
    seed_incl = stats.get("phase.seed", {}).get("inclusive", 0.0)
    panel_incl = stats.get("data.load_panel", {}).get("self", 0.0)
    n_candidates = len(EVAL_SAMPLES)
    fits = int(stats.get("eval._combined_prediction", {}).get("count", 0))

    # eval self-time table, sorted, additive
    eval_rows = []
    eval_self_sum = 0.0
    for label, rec in stats.items():
        if not label.startswith("eval."):
            continue
        eval_self_sum += rec["self"]
        eval_rows.append({
            "metric": label[len("eval."):],
            "self_sec": rec["self"],
            "inclusive_sec": rec["inclusive"],
            "count": int(rec["count"]),
            "mean_ms": (rec["inclusive"] / rec["count"] * 1000) if rec["count"] else 0.0,
        })
    eval_rows.sort(key=lambda r: r["self_sec"], reverse=True)

    # book-size scaling buckets
    buckets: dict[str, list[float]] = {}
    for book_size, dt in EVAL_SAMPLES:
        key = ("00-04" if book_size < 5 else "05-09" if book_size < 10
               else "10-19" if book_size < 20 else "20-39" if book_size < 40
               else "40+")
        buckets.setdefault(key, []).append(dt)
    scaling = {k: {"n": len(v), "mean_sec": sum(v)/len(v)}
               for k, v in sorted(buckets.items())}

    return {
        "config": vars(args),
        "wall_sec": wall,
        "phases": {
            "panel_load_sec": panel_incl,
            "seed_sec": seed_incl,
            "evaluate_total_sec": eval_incl,
            "generations": summary.get("generations"),
            "n_trials": summary.get("n_trials"),
            "n_eval_failures": summary.get("n_eval_failures"),
            "archive_size": len(summary.get("archive", [])),
        },
        "llm": {
            "total_cost_usd": meter.total_cost(),
            "budget_usd": args.max_cost_usd,
            "total_calls": meter.total_calls(),
            "by_role": meter.by_role,
        },
        "evaluation": {
            "n_candidates_scored": n_candidates,
            "model_fits_total": fits,
            "model_fits_per_candidate": (fits / n_candidates) if n_candidates else 0.0,
            "mean_eval_sec_per_candidate": (eval_incl / n_candidates) if n_candidates else 0.0,
            "self_time_sum_sec": eval_self_sum,
            "metrics": eval_rows,
            "scaling_by_book_size": scaling,
        },
    }


def print_report(rep: dict[str, Any]) -> None:
    p, llm, ev = rep["phases"], rep["llm"], rep["evaluation"]
    line = "=" * 78
    print("\n" + line)
    print("EVOLUTION TIMING DIAGNOSTIC")
    print(line)
    print(f"Wall clock ............. {_fmt(rep['wall_sec'])}")
    print(f"Candidates scored ...... {ev['n_candidates_scored']}  "
          f"(generations={p['generations']}, n_trials={p['n_trials']}, "
          f"eval-failures={p['n_eval_failures']}, archive={p['archive_size']})")

    print(f"\n── PHASES (inclusive wall) {'─'*51}")
    print(f"  panel load (once) .... {_fmt(p['panel_load_sec'])}")
    print(f"  seeding .............. {_fmt(p['seed_sec'])}")
    print(f"  evaluation (total) ... {_fmt(p['evaluate_total_sec'])}  "
          f"← dominant compute")

    print(f"\n── LLM (gpt-4o-mini) {'─'*57}")
    print(f"  {'role':<12}{'calls':>7}{'time':>10}{'in_tok':>10}"
          f"{'out_tok':>10}{'cost $':>10}")
    for role, r in sorted(llm["by_role"].items(),
                          key=lambda kv: kv[1]["cost"], reverse=True):
        print(f"  {role:<12}{int(r['calls']):>7}{_fmt(r['seconds']):>10}"
              f"{int(r['input_tokens']):>10}{int(r['output_tokens']):>10}"
              f"{r['cost']:>10.4f}")
    print(f"  {'TOTAL':<12}{llm['total_calls']:>7}{'':>10}{'':>10}{'':>10}"
          f"{llm['total_cost_usd']:>10.4f}")
    print(f"  budget ${llm['budget_usd']:.2f}  →  used "
          f"{100*llm['total_cost_usd']/llm['budget_usd']:.1f}% of budget")

    print(f"\n── EVALUATION METRICS (self-time, additive) {'─'*34}")
    print(f"  model fits total: {ev['model_fits_total']}  "
          f"(~{ev['model_fits_per_candidate']:.1f}/candidate)   "
          f"mean {_fmt(ev['mean_eval_sec_per_candidate'])}/candidate")
    print(f"  {'metric':<26}{'self':>10}{'incl':>10}{'calls':>9}{'ms/call':>10}")
    for r in ev["metrics"]:
        print(f"  {r['metric']:<26}{_fmt(r['self_sec']):>10}"
              f"{_fmt(r['inclusive_sec']):>10}{r['count']:>9}"
              f"{r['mean_ms']:>10.1f}")
    print(f"  {'(self-time sum)':<26}{_fmt(ev['self_time_sum_sec']):>10}")

    print(f"\n── EVAL COST vs BOOK SIZE (the O(book_size) blowup) {'─'*26}")
    print(f"  {'book size':<12}{'n':>6}{'mean eval':>12}")
    for k, v in ev["scaling_by_book_size"].items():
        print(f"  {k:<12}{v['n']:>6}{_fmt(v['mean_sec']):>12}")
    print(line + "\n")


def print_extrapolation(rep: dict[str, Any]) -> None:
    ev = rep["evaluation"]
    mean = ev["mean_eval_sec_per_candidate"]
    if mean <= 0:
        return
    print("── EXTRAPOLATION ─────────────────────────────────────────────────")
    print(f"  This run: {ev['n_candidates_scored']} candidates × "
          f"{_fmt(mean)}/candidate ≈ {_fmt(mean*ev['n_candidates_scored'])} eval.")
    print("  Evaluation time scales with (#candidates × folds × book_size) and")
    print("  with the panel row count (n_tickers × dev_bars). The 24h GP run had")
    print("  ~203 candidates on the FULL S&P100 (~105 tickers) with the archive")
    print("  growing to ~57 — both the candidate count and the per-candidate")
    print("  matrix were far larger than this reduced-ticker diagnostic.")
    print("  Levers the numbers above will price out: fewer CPCV folds, a linear")
    print("  --marginal-model, parallel folds, freezing the book within a gen.\n")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config-file", default="quant.config.sp100.yaml",
                   help="Data config to activate (sets QF_CONFIG_FILE).")
    p.add_argument("--config-name", default=None,
                   help="Workspace scope name (default: derived from the config).")
    p.add_argument("--name", default="timing_diag", help="Prerun id within the scope.")
    p.add_argument("--n-tickers", type=int, default=15)
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--population", type=int, default=8)
    p.add_argument("--children-per-gen", type=int, default=6)
    p.add_argument("--seed-ideas", type=int, default=6)
    p.add_argument("--seed-papers", type=int, default=0)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--stability-blocks", type=int, default=4)
    p.add_argument("--marginal-model", default="gradient_boosting",
                   help="Combiner for the marginal (LOCO) fits (try 'ridge' to "
                        "price the nonlinear-model cost).")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--max-cost-usd", type=float, default=3.0,
                   help="Hard cost ceiling; the run aborts if exceeded.")
    p.add_argument("--no-persist", action="store_true",
                   help="Skip the final materialise/persist phase.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── env, BEFORE importing quant_fund_agent ──
    load_dotenv()
    os.environ["FACTOR_RESEARCH_LLM_MODEL"] = args.model
    os.environ.setdefault("FACTOR_RESEARCH_LLM_PROVIDER", "openai")
    os.environ["QF_USE_MCP"] = "0"           # in-process eval → patches apply
    os.environ["RESEARCH_USE_MCP"] = "0"
    os.environ["QF_CONFIG_FILE"] = args.config_file

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)-20s  %(message)s",
        datefmt="%H:%M:%S")
    log = logging.getLogger("timing")

    meter = LLMMeter(max_cost_usd=args.max_cost_usd)
    install_instrumentation(meter)

    from quant_fund_agent.agents.factor_research.evolution.loop import (
        EvolutionLoop, EvolutionRunConfig, persist_archive,
    )
    from quant_fund_agent.config import default_config_name, get_settings
    from quant_fund_agent.workspace import Scope

    settings = get_settings()
    config_name = args.config_name or default_config_name(settings.data)
    scope = Scope(config_name, args.name)
    log.info("Diagnostic scope '%s' (model=%s, config=%s)",
             scope.label, args.model, args.config_file)
    scope.purge()
    scope.ensure()
    scope.write_config_snapshot(settings.data)
    os.environ["FACTOR_DB_PATH"] = str(scope.factor_db_path)
    os.environ["PAPER_READ_LOG"] = str(scope.read_log_path)
    os.environ["QF_SCOPE"] = scope.label

    cfg = EvolutionRunConfig(
        generations=args.generations,
        population_size=args.population,
        children_per_generation=args.children_per_gen,
        n_seed_ideas=args.seed_ideas,
        seed_papers=args.seed_papers,
        target_horizon=args.horizon,
        stability_blocks=args.stability_blocks,
        marginal_model=args.marginal_model,
        n_tickers=args.n_tickers,
        out_dir=str(scope.dir / "evolution"),
    )

    wall0 = time.perf_counter()
    try:
        loop = EvolutionLoop(cfg)
        summary = loop.run()
        if not args.no_persist:
            with TIMER.span("phase.persist"):
                persist_archive(
                    loop.controller,
                    session_id=f"timing:{config_name}:{args.name}",
                    target_horizon=args.horizon, n_tickers=args.n_tickers,
                    is_frac=cfg.is_frac, val_frac=cfg.val_frac, fields=loop.fields,
                    marginal_model=args.marginal_model)
    except RuntimeError as e:
        log.error("%s", e)
        summary = {"generations": None, "n_trials": None, "archive": [],
                   "n_eval_failures": None, "aborted": True}
    wall = time.perf_counter() - wall0

    rep = build_report(meter, wall, summary, args)
    print_report(rep)
    print_extrapolation(rep)

    out_dir = Path("data/diagnostics") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "timing.json").write_text(json.dumps(rep, indent=2, default=str))
    (out_dir / "report.md").write_text(_markdown(rep))
    print(f"Wrote {out_dir/'timing.json'}\n      {out_dir/'report.md'}")


def _markdown(rep: dict[str, Any]) -> str:
    p, llm, ev = rep["phases"], rep["llm"], rep["evaluation"]
    lines = [
        "# Evolution timing diagnostic", "",
        f"- Wall clock: **{_fmt(rep['wall_sec'])}**",
        f"- Candidates scored: **{ev['n_candidates_scored']}** "
        f"(generations={p['generations']}, n_trials={p['n_trials']}, "
        f"archive={p['archive_size']})",
        f"- Model fits: **{ev['model_fits_total']}** "
        f"(~{ev['model_fits_per_candidate']:.1f}/candidate)",
        f"- LLM cost: **${llm['total_cost_usd']:.4f}** of ${llm['budget_usd']:.2f} "
        f"budget ({llm['total_calls']} calls)", "",
        "## Phases", "",
        f"| phase | time |", "|---|---|",
        f"| panel load (once) | {_fmt(p['panel_load_sec'])} |",
        f"| seeding | {_fmt(p['seed_sec'])} |",
        f"| evaluation (total) | {_fmt(p['evaluate_total_sec'])} |", "",
        "## LLM by role", "",
        "| role | calls | time | in_tok | out_tok | cost $ |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for role, r in sorted(llm["by_role"].items(),
                          key=lambda kv: kv[1]["cost"], reverse=True):
        lines.append(f"| {role} | {int(r['calls'])} | {_fmt(r['seconds'])} "
                     f"| {int(r['input_tokens'])} | {int(r['output_tokens'])} "
                     f"| {r['cost']:.4f} |")
    lines += ["", "## Evaluation metrics (self-time, additive)", "",
              "| metric | self | inclusive | calls | ms/call |",
              "|---|--:|--:|--:|--:|"]
    for r in ev["metrics"]:
        lines.append(f"| {r['metric']} | {_fmt(r['self_sec'])} "
                     f"| {_fmt(r['inclusive_sec'])} | {r['count']} "
                     f"| {r['mean_ms']:.1f} |")
    lines += ["", "## Eval cost vs book size", "",
              "| book size | n | mean eval |", "|---|--:|--:|"]
    for k, v in ev["scaling_by_book_size"].items():
        lines.append(f"| {k} | {v['n']} | {_fmt(v['mean_sec'])} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
