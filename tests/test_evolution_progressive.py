"""Progressive data reveal — schedule math, MCP-seam threading, loop integration.

Split into four groups following the plan's test spec:
1. schedule math (pure, on a synthetic DatetimeIndex);
2. seam threading (in-process ``QF_USE_MCP=0``): calendar split + signal-cache windowing;
3. loop integration: frontier advances, archive re-scored without n_trials billing,
   prequential log, dedup retry;
4. OFF-mode invariance + resume determinism.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.agents.factor_research.evolution.progressive import (
    MIN_IS_BARS,
    GenerationWindow,
    build_schedule,
    reveal_generations,
)


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="D")


# ── group 1: schedule math ─────────────────────────────────────────────────────

def test_schedule_seed_block_boundaries_and_reaches_dev_end():
    idx = _index(1000)
    sched = build_schedule(idx, generations=4, test_frac=0.2, seed_frac=0.5,
                           reveal_every=1, val_blocks=2)
    # one window per generation 0..G
    assert [w.generation for w in sched] == [0, 1, 2, 3, 4]
    # dev_len = 800, seed_end = 400, R = 4 reveals, block_len = 100, val_span = 200
    assert sched[0].reveal is False and sched[0].block_bounds is None
    assert sched[0].visible_end == 400 and sched[0].val_start == 200
    assert [w.visible_end for w in sched] == [400, 500, 600, 700, 800]
    assert [w.val_start for w in sched] == [200, 300, 400, 500, 600]
    # every g≥1 is a reveal generation and carries its block bounds
    assert all(w.reveal for w in sched[1:])
    assert sched[1].block_bounds == (idx[400].isoformat(), idx[500].isoformat())
    # the last reveal reaches exactly the dev end (test_start = 800)
    assert sched[-1].visible_end == 800
    # timestamps follow the three_way_split calendar convention
    assert sched[2].is_end_ts == idx[400].isoformat()
    assert sched[2].val_end_ts == idx[600].isoformat()


def test_test_tail_is_never_inside_any_window():
    idx = _index(1000)
    test_start = 800  # floor(1000 * (1 - 0.2))
    sched = build_schedule(idx, generations=4, test_frac=0.2, seed_frac=0.5,
                           reveal_every=1, val_blocks=2)
    for w in sched:
        # VAL is [is_end, val_end); val_end == index[visible_end] is EXCLUSIVE, so no
        # scored bar is ≥ test_start
        assert w.visible_end <= test_start
        assert pd.Timestamp(w.val_end_ts) <= idx[test_start]


def test_reveal_every_two_and_remainder_lands_in_last_block():
    idx = _index(1000)
    # generations=3, reveal_every=1 → R=3; remaining=400; block_len=133, remainder 1
    sched = build_schedule(idx, generations=3, test_frac=0.2, seed_frac=0.5,
                           reveal_every=1, val_blocks=1)
    ends = [w.visible_end for w in sched]
    assert ends == [400, 533, 666, 800]   # gen0 = seed_end; last block absorbs remainder
    assert ends[-1] - ends[-2] == 134     # 800 - 666, one bar bigger than a 133 block
    assert (ends[-1] - ends[-2]) > (ends[-2] - ends[-3])

    # reveal_every=2 over 4 generations → reveal on g=1,3 only
    assert reveal_generations(4, 2) == [1, 3]
    sched2 = build_schedule(idx, generations=4, test_frac=0.2, seed_frac=0.5,
                            reveal_every=2, val_blocks=1)
    assert [w.reveal for w in sched2] == [False, True, False, True, False]
    # non-reveal generations keep the previous frontier
    assert sched2[1].visible_end == sched2[2].visible_end
    assert sched2[3].visible_end == sched2[4].visible_end


def test_is_clamp_keeps_thirty_bars_and_shrinks_val():
    idx = _index(200)
    # dev_len=160, seed_end=40, R=2, block_len=60, val_span=120 → gen0/1 clamp binds
    sched = build_schedule(idx, generations=2, test_frac=0.2, seed_frac=0.25,
                           reveal_every=1, val_blocks=2)
    assert sched[0].val_start == MIN_IS_BARS        # clamped up to 30 (IS kept ≥ 30)
    assert sched[1].val_start == MIN_IS_BARS
    assert sched[2].val_start == 40                 # 160 - 120, no clamp needed
    # VAL shrinks under the clamp but stays non-empty
    assert sched[0].visible_end - sched[0].val_start >= 1


def test_schedule_is_deterministic():
    idx = _index(1000)
    kw = dict(generations=4, test_frac=0.2, seed_frac=0.5, reveal_every=1, val_blocks=2)
    assert build_schedule(idx, **kw) == build_schedule(idx, **kw)


def test_schedule_config_validation_errors():
    idx = _index(1000)
    base = dict(generations=4, test_frac=0.2, seed_frac=0.5, reveal_every=1, val_blocks=2)
    for bad in (
        {"test_frac": 0.0}, {"test_frac": 1.0},
        {"seed_frac": 0.0}, {"seed_frac": 1.0},
        {"reveal_every": 0}, {"val_blocks": 0}, {"generations": 0},
    ):
        with pytest.raises(ValueError):
            build_schedule(idx, **{**base, **bad})
    # block_len < 1: seed eats the whole dev window
    with pytest.raises(ValueError):
        build_schedule(_index(100), generations=50, test_frac=0.2, seed_frac=0.99,
                       reveal_every=1, val_blocks=1)
    # generation-0 degenerate: too little dev to keep 30 IS + a VAL bar
    with pytest.raises(ValueError):
        build_schedule(_index(80), generations=2, test_frac=0.2, seed_frac=0.3,
                       reveal_every=1, val_blocks=4)


# ── group 2: MCP-seam threading (in-process) ───────────────────────────────────

N_BARS, TICKERS = 480, ["A", "B", "C", "D", "E", "F"]

MOM_BODY = "close.pct_change().fillna(0.0)"
SMOOTH_BODY = "close.pct_change().rolling(3).mean().fillna(0.0)"
CHILD_BODY = "close.pct_change().rolling(2).mean().fillna(0.0)"


def _factor_code(fid: str, body: str = MOM_BODY) -> str:
    return f'''\
"""Test factor {fid}."""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor


@register_factor
class F_{fid}(BaseFactor):
    factor_id = "{fid}"
    name = "{fid}"
    category = "momentum"
    inputs = ["close"]
    prediction_horizon = 1

    def calc(self, data):
        close = data["close"]
        return {body}
'''


@pytest.fixture()
def synth_panel():
    rng = np.random.default_rng(3)
    rets = np.zeros((N_BARS, len(TICKERS)))
    eps = rng.standard_normal((N_BARS, len(TICKERS))) * 0.01
    for t in range(1, N_BARS):
        rets[t] = 0.6 * rets[t - 1] + eps[t]
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="D")
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=TICKERS)
    return {"close": close}


@pytest.fixture()
def wired_service(synth_panel, monkeypatch):
    monkeypatch.setenv("QF_USE_MCP", "0")
    from quant_fund_agent.mcp import research_service as svc

    monkeypatch.setattr(svc, "_load_panel_cached",
                        lambda data_dir, fields, n_tickers: synth_panel)
    monkeypatch.setattr(svc, "_panel_cache_key",
                        lambda data_dir, fields, n_tickers: ("prog-panel",))
    svc._SIGNAL_CACHE.clear()
    return svc, synth_panel


def test_calendar_split_sizes_match_bounds(wired_service):
    svc, panel = wired_service
    idx = panel["close"].index
    is_end, val_end = idx[240].isoformat(), idx[360].isoformat()
    res = svc.evaluate_fitness(
        {"factor_id": "cand", "code": _factor_code("cand"), "expected_sign": 1},
        target_horizon=1, is_end=is_end, val_end=val_end, fields=["close"],
        n_tickers=None,
    )
    assert res["ok"], res.get("error")
    # the held-out (full) split reflects the calendar bounds exactly
    assert res["fitness"]["raw"]["heldout_test_split_sizes"] == {
        "is": 240, "val": 120, "test": 120}


def test_signal_cache_is_window_keyed(wired_service):
    svc, panel = wired_service
    idx = panel["close"].index
    prog = {"factor_id": "cand", "code": _factor_code("cand"), "expected_sign": 1}
    common = dict(target_horizon=1, fields=["close"], n_tickers=None)

    r1 = svc.evaluate_fitness(prog, is_end=idx[200].isoformat(),
                              val_end=idx[300].isoformat(), **common)
    r2 = svc.evaluate_fitness(prog, is_end=idx[300].isoformat(),
                              val_end=idx[420].isoformat(), **common)
    assert r1["ok"] and r2["ok"]
    # the second, longer frontier is scored on a strictly larger IS∪VAL window — a
    # stale short-window signal was NOT reused
    dev1 = r1["fitness"]["raw"]["split_sizes"]
    dev2 = r2["fitness"]["raw"]["split_sizes"]
    assert dev1["is"] + dev1["val"] == 300
    assert dev2["is"] + dev2["val"] == 420
    # the cache holds a distinct entry per frontier for the same program fingerprint
    frontiers = {k[2] for k in svc._SIGNAL_CACHE}
    assert idx[300].isoformat() in frontiers and idx[420].isoformat() in frontiers


# ── group 3: loop integration ──────────────────────────────────────────────────


class _FakeLLM:
    """Deterministic mutating LLM: always emits one valid child factor program."""

    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        payload = {
            "factor_id": "evo_child",
            "name": "Evo child",
            "category": "momentum",
            "trading_idea": "smoothed momentum persists",
            "description": "smoothed momentum",
            "prediction_horizon": 1,
            "suggested_horizons": [1, 3],
            "expected_sign": 1,
            "code": _factor_code("evo_child", CHILD_BODY),
        }

        class _Resp:
            content = json.dumps(payload)

        return _Resp()


def _seeds():
    from quant_fund_agent.agents.factor_research.evolution.genome import FactorProgram

    return [
        FactorProgram(factor_id="seed_mom", code=_factor_code("seed_mom", MOM_BODY),
                      name="seed momentum", trading_idea="momentum", expected_sign=1,
                      prediction_horizon=1),
        FactorProgram(factor_id="seed_smooth",
                      code=_factor_code("seed_smooth", SMOOTH_BODY),
                      name="seed smooth", trading_idea="smoothed momentum",
                      expected_sign=1, prediction_horizon=1),
    ]


@pytest.fixture()
def prog_loop(synth_panel, tmp_path, monkeypatch):
    """An EvolutionLoop wired to the synthetic panel + fake LLM, progressive ON."""
    monkeypatch.setenv("QF_USE_MCP", "0")
    from quant_fund_agent.agents.factor_research.evolution import loop as loop_mod
    from quant_fund_agent.agents.factor_research.evolution.loop import (
        EvolutionLoop,
        EvolutionRunConfig,
    )
    from quant_fund_agent.mcp import research_service as svc

    monkeypatch.setattr(svc, "_load_panel_cached",
                        lambda data_dir, fields, n_tickers: synth_panel)
    monkeypatch.setattr(svc, "_panel_cache_key",
                        lambda data_dir, fields, n_tickers: ("prog-panel",))
    svc._SIGNAL_CACHE.clear()
    fake = _FakeLLM()
    monkeypatch.setattr(loop_mod, "_get_llm", lambda temperature, role=None: fake)

    def _make(**overrides):
        params = dict(
            generations=2, population_size=6, children_per_generation=4,
            seed=11, target_horizon=1, n_tickers=None,
            progressive=True, test_frac=0.2, seed_frac=0.5, reveal_every=1,
            val_blocks=1, out_dir=str(tmp_path / "evolution"),
        )
        params.update(overrides)
        cfg = EvolutionRunConfig(**params)
        loop = EvolutionLoop(cfg, data_context="CTX",
                             fields=["open", "high", "low", "close", "volume"])
        loop._load_known_ids = lambda: None
        return loop

    return _make, tmp_path


def test_progressive_loop_advances_frontier_and_logs_prequential(prog_loop):
    make, tmp_path = prog_loop
    loop = make()
    summary = loop.run(initial_programs=_seeds())
    assert summary["generations"] == 2

    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="D")
    # dev_len=384, seed_end=192, block_len=96 → reveals at index[192]→288, 288→384
    prequential = tmp_path / "evolution" / "prequential.jsonl"
    assert prequential.exists()
    rows = [json.loads(l) for l in prequential.read_text().splitlines()]
    assert [r["generation"] for r in rows] == [1, 2]
    scored = [r for r in rows if "skipped" not in r]
    assert scored, "archive should be non-empty so prequential actually scored"
    for r in scored:
        # [start, end) equals the newly revealed block bounds, end ≤ dev boundary
        assert pd.Timestamp(r["end"]) <= idx[384]
        assert pd.Timestamp(r["start"]) < pd.Timestamp(r["end"])
    starts_ends = {(r["start"], r["end"]) for r in rows}
    assert (idx[192].isoformat(), idx[288].isoformat()) in starts_ends
    assert (idx[288].isoformat(), idx[384].isoformat()) in starts_ends
    # reveal stamps: position among the reveal generations + the new frontier
    assert [r["reveal_index"] for r in rows] == [0, 1]
    for r in rows:
        assert r["frontier_ts"] == r["end"]
        assert isinstance(r["visible_end"], int)
    assert [r["visible_end"] for r in rows] == [288, 384]


def test_prequential_rows_are_deduped_on_rewrite(prog_loop):
    """A generation that already has a prequential row is never written twice
    (the crashed-then-resumed run re-enters _advance_reveal for the same gen)."""
    make, tmp_path = prog_loop
    loop = make()
    loop.run(initial_programs=_seeds())
    preq = tmp_path / "evolution" / "prequential.jsonl"
    n_rows = len(preq.read_text().splitlines())

    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="D")
    # same-instance retry is skipped …
    loop._prequential_score(1, idx[192].isoformat(), idx[288].isoformat())
    assert len(preq.read_text().splitlines()) == n_rows
    # … and so is a fresh instance that lazily reloads the dedup set from disk
    loop._prequential_done = None
    loop._prequential_score(2, idx[288].isoformat(), idx[384].isoformat())
    assert len(preq.read_text().splitlines()) == n_rows
    # a genuinely new generation still appends
    loop._prequential_score(99, idx[288].isoformat(), idx[384].isoformat())
    rows = [json.loads(l) for l in preq.read_text().splitlines()]
    assert len(rows) == n_rows + 1 and rows[-1]["generation"] == 99


def test_rescore_does_not_bill_ntrials_and_stamps_window(prog_loop):
    make, _ = prog_loop
    loop = make()
    loop.run(initial_programs=_seeds())

    lineage = loop.controller.lineage
    rescore_rows = [r for r in lineage if r.get("event") == "rescore"]
    billed_rows = [r for r in lineage if "event" not in r]
    # re-scores happened on the reveal generations …
    assert rescore_rows
    assert all("objective_before" in r and "objective_after" in r for r in rescore_rows)
    # … each carrying the reveal stamps + a diagnostics sub-dict
    for r in rescore_rows:
        assert r["reveal_index"] is not None
        assert r["scored_through"] is not None
        assert {"val_ic", "marginal_value_raw", "coverage"} <= set(r["diagnostics"])
    # … but did NOT increment the multiple-testing budget (one trial per billed insert)
    assert loop.controller.n_trials == len(billed_rows)
    # every scored fitness records the window it was measured through
    assert loop.controller.archive
    for eg in loop.controller.archive:
        assert eg.fitness.diagnostics.get("scored_through") is not None
        assert eg.fitness.diagnostics.get("window_generation") is not None


def test_failed_rescore_emits_visible_lineage_row(prog_loop):
    """A rescore that errors out appends a rescore_failed row (stale fitness kept)."""
    make, _ = prog_loop
    loop = make()
    loop.run(initial_programs=_seeds())
    assert loop.controller.archive

    fitness_before = {eg.genome.genome_id: eg.fitness.objective.to_dict()
                      for eg in loop.controller.archive}
    loop._score_program = lambda *a, **kw: {"ok": False, "error": "boom"}
    loop._score_set = lambda *a, **kw: {"ok": False, "error": "boom"}
    loop._rescore_archive_on_window(2)

    failed = [r for r in loop.controller.lineage
              if r.get("event") == "rescore_failed"]
    assert failed and all(r["error"] == "boom" for r in failed)
    assert {"generation", "genome_id", "error"} <= set(failed[0])
    # behaviour otherwise unchanged: stale fitness kept, members not dropped
    for eg in loop.controller.archive:
        assert eg.fitness.objective.to_dict() == fitness_before[eg.genome.genome_id]


def test_gate_failer_fingerprint_is_retriable_once():
    from quant_fund_agent.agents.factor_research.evolution.controller import (
        EvaluatedGenome,
        EvolutionController,
    )
    from quant_fund_agent.agents.factor_research.evolution.genome import (
        FactorProgram,
        Genome,
    )
    from quant_fund_agent.research_eval.fitness import (
        FitnessResult,
        GateResults,
        ObjectiveVector,
    )

    ctrl = EvolutionController()
    genome = Genome(genome_id="g0-fail",
                    programs=[FactorProgram(factor_id="f", code="x = 1\n")])
    failing = FitnessResult(
        candidate_id="f",
        objective=ObjectiveVector(marginal_value=0.0),
        gates=GateResults(coverage_ok=False))       # gate-failing
    ctrl.insert(EvaluatedGenome(genome=genome, fitness=failing))

    assert ctrl.is_duplicate(genome)                # deduped after the first look
    assert ctrl.release_failed_fingerprints() == 1  # freed once on a reveal
    assert not ctrl.is_duplicate(genome)            # now re-evaluable
    ctrl.insert(EvaluatedGenome(genome=genome, fitness=failing))  # fails again
    assert ctrl.release_failed_fingerprints() == 0  # capped: never a second retry


# ── group 4: OFF-mode invariance ───────────────────────────────────────────────

def test_off_mode_calls_client_with_no_calendar_bounds(synth_panel, tmp_path,
                                                        monkeypatch):
    monkeypatch.setenv("QF_USE_MCP", "0")
    from quant_fund_agent.agents.factor_research.evolution.genome import FactorProgram
    from quant_fund_agent.agents.factor_research.evolution.loop import (
        EvolutionLoop,
        EvolutionRunConfig,
    )
    from quant_fund_agent.mcp import research_client

    captured: dict = {}

    def fake_eval(candidate, book=None, jitter=None, reference=None, **kwargs):
        captured.update(kwargs)
        from quant_fund_agent.research_eval.fitness import (
            FitnessResult,
            GateResults,
            ObjectiveVector,
        )
        fit = FitnessResult(candidate_id=candidate["factor_id"],
                            objective=ObjectiveVector(marginal_value=0.1),
                            gates=GateResults(coverage_ok=True),
                            diagnostics={"n_trials": kwargs["n_trials"]})
        return {"ok": True, "fitness": fit.to_dict()}

    monkeypatch.setattr(research_client, "evaluate_fitness", fake_eval)
    cfg = EvolutionRunConfig(target_horizon=1, progressive=False,
                             out_dir=str(tmp_path / "evolution"))
    loop = EvolutionLoop(cfg, data_context="CTX", fields=["close"])
    loop.evaluate_program(FactorProgram(factor_id="c", code=_factor_code("c")))

    # legacy mode threads NO calendar bounds — the harness uses is_frac/val_frac
    assert captured["is_end"] is None and captured["val_end"] is None
    assert loop._window_bounds == (None, None)


# ── group 5: resume ────────────────────────────────────────────────────────────

def test_resume_recomputes_frontier_and_continues(prog_loop):
    from quant_fund_agent.agents.factor_research.evolution.controller import (
        EvolutionController,
    )

    make, tmp_path = prog_loop
    # a genuine mid-run crash: run generations=3 but raise before the gen-2 checkpoint,
    # so state.json is left cleanly at generation 1 (the schedule is still built for G=3).
    first = make(generations=3)
    orig_ckpt = first._checkpoint

    def crash_before_gen2_checkpoint():
        if first.controller.generation >= 2:
            raise RuntimeError("simulated crash")
        orig_ckpt()

    first._checkpoint = crash_before_gen2_checkpoint
    with pytest.raises(RuntimeError):
        first.run(initial_programs=_seeds())

    state = tmp_path / "evolution" / "state.json"
    assert state.exists()
    reloaded = EvolutionController.load(state)
    assert reloaded.generation == 1                 # last clean checkpoint
    assert isinstance(reloaded._failed_fingerprints, dict)  # round-trips

    # the last clean checkpoint's lineage history (what resume must preserve)
    lineage_at_crash = [json.loads(l) for l in
                        (tmp_path / "evolution" / "lineage.jsonl")
                        .read_text().splitlines()]
    assert lineage_at_crash

    # resume from the loaded controller with the SAME generations → guard must pass
    resumed = make(generations=3)
    resumed.controller = reloaded
    summary = resumed.run(resume=True)
    assert summary["generations"] == 3
    assert resumed.controller.generation == 3
    # the recomputed schedule matches the crashed run's (pure fn of config + index)
    assert [w.val_end_ts for w in resumed._schedule] == \
        [w.val_end_ts for w in first._schedule]

    # lineage history was reloaded, not truncated: the pre-crash rows are still
    # the head of the rewritten file (incl. the generation-0 seed rows)
    lineage_after = [json.loads(l) for l in
                     (tmp_path / "evolution" / "lineage.jsonl")
                     .read_text().splitlines()]
    assert lineage_after[:len(lineage_at_crash)] == lineage_at_crash
    assert len(lineage_after) > len(lineage_at_crash)
    assert any("event" not in r and r["generation"] == 0 and r["operator"] == "seed"
               for r in lineage_after)

    # prequential + gen_quality rows are deduped across the crash/resume boundary
    preq_gens = [json.loads(l)["generation"] for l in
                 (tmp_path / "evolution" / "prequential.jsonl")
                 .read_text().splitlines()]
    assert preq_gens == sorted(set(preq_gens))
    gq_gens = [json.loads(l)["generation"] for l in
               (tmp_path / "evolution" / "gen_quality.jsonl")
               .read_text().splitlines()]
    assert gq_gens == sorted(set(gq_gens))
