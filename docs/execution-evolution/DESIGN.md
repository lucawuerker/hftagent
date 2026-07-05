# Evolutionary Execution Researcher — Design & Architecture

Updated: 2026-07-05 (accordance pass vs the factor-evolution leak-tightening
changes of 2026-07-04/05 — see §Leak-free evaluation conventions, inherited)
Status: **AGREED — ready to build (nothing implemented yet).** All design
forks were resolved with the author on 2026-07-03 (see §Locked decisions and
§Resolved questions). This document is the implementation-anchor *prompt* for
the build — the same role `docs/research-evolution/DESIGN.md` played for the
factor researcher; a companion `IMPLEMENTATION_PROGRESS.md` will track the
phases once the build starts.

## Purpose

The factor researcher now *evolves alpha*; everything downstream of the alpha
is still hand-written. Concretely, in honest terms:

* The **Architect** is a menu-picker: the LLM chooses a model type from a fixed
  catalog, hyper-parameters, a factor subset and position-construction
  *settings* — it never invents anything.
* The **signal → positions mapping is hardcoded** in two divergent pipelines:
  cross-sectional (`backtesting/strategy_backtester._signal_to_positions`:
  winsorise → z-score → top-`max_positions` → dollar-neutral rescale) and
  per-underlying (`backtesting/positions.py`: z-basis → threshold/sign/
  continuous → `1/max_positions` equal weight), with a third near-copy inside
  the comparison harness. No stops, no volatility targeting, no
  drawdown-aware de-risking, no signal-decay-aware holding — none of the
  execution intelligence a real fund has.
* The **PM** picks from seven fixed allocators.

This document designs the next enhancement: an **evolutionary Execution
Researcher** that applies the FunSearch/AlphaEvolve methodology one level up
the stack — *"first evolve the alpha, then evolve how you trade the alpha."*
The LLM is again the mutation operator, this time over **execution programs**
(the code that turns a strategy's composite signal into a target book through
time), and a deterministic, cost-aware, walk-forward harness is again the
fitness function. The thesis contribution stays *agent methodology*: one
search paradigm demonstrated at two levels of a quant fund's stack.

### Why program synthesis and not the alternatives (surveyed 2026-07)

Alternatives considered with the author and set aside for the core build:

* **LLM-in-the-loop trading** (FinMem/FinAgent style — the LLM decides at each
  bar/meeting): non-reproducible, expensive per bar, and it violates the locked
  principle that the LLM must never influence its own reward. Rejected for any
  runtime use (locked decision 8) — the LLM's agency lives entirely at research
  time.
* **Config-genome AutoML** (evolve model+hyperparams+settings vectors): cheap
  but menu-bound — the Architect already does a weaker LLM version; least
  defensible as a contribution.
* **ADAS-style meta-agent search / DSPy**: optimise the agent pipeline or its
  prompts, not the trading logic; no finance-grade overfitting control.
  Orthogonal; may be borrowed later for prompt tuning.
* **Voyager-style skill library**: *borrowed* — the Pareto archive doubles as
  a growing library of named, reusable execution skills (see §Archive).

**Decisions locked with the author (2026-07-03):**

1. **Target = position/execution programs first** (the signal→book function).
   Full strategy programs (entry/exit logic around the model) are the natural
   phase-2 extension; PM-policy evolution is deferred indefinitely (too few
   strategies → no statistical power to score policies).
2. **Agency at research time, determinism at runtime.** The LLM invents and
   mutates execution programs during research; deployment runs the winning
   program deterministically. (A periodic LLM "regime overlay" that switches
   between archived programs is an optional later ablation, not the core.)
3. **New engine alongside, never a replacement.** Today's Architect and the
   two hardcoded pipelines stay as the baseline arm; the evolutionary engine
   is a parallel path, exactly like `oneshot` vs `evolution` for the factor
   researcher.
4. **Primary fitness = net-of-cost OOS Sharpe**, deflated and walk-forward
   (secondary axes and hard gates below).
5. **Cost model = the existing spread-aware layer.** It is the reward, held
   identical across candidates; the ±50% cost-sensitivity re-score (reported
   as a diagnostic) exposes cost-model gaming. A participation-based impact
   upgrade is a later, orthogonal improvement.
6. **Cross-signal axis uses 3–5 diverse frozen signals** from existing
   strategies/preruns — different model families (e.g. ridge vs GBM) *and*
   different factor subsets, fixed once per run.
7. **State scope = core set + book state.** Executors condition on trailing
   vol, ADV/liquidity, spread (where the feed carries it), running strategy
   drawdown, signal age, **and the current book (positions + unrealised P&L
   per name)** — enabling true path-dependent logic (stops, profit-taking,
   position-aware rebalancing). This makes the canonical contract *stepwise*
   (see §Genome). Market-regime flags (index trend, dispersion) are a later
   additive extension, not in v1.
8. **No LLM at runtime, period.** The earlier idea of an LLM "regime overlay"
   switching between archived executors at PM meetings is **dropped** (not
   even as an ablation arm): the deployed system is fully deterministic.

## The single most important principle (unchanged)

> **The LLM ideates and mutates; a deterministic harness scores. The LLM must
> never be able to influence its own reward.**

Everything from `research-evolution/DESIGN.md` about the two channels
(selection fitness = cold numbers → controller; diagnostic feedback = rich NL
→ the mutating LLM) carries over verbatim. The execution layer adds one new
temptation to guard against: an execution program can "win" by exploiting
**backtest artifacts** (bar-boundary effects, the cost model's blind spots,
non-causal standardisation) rather than genuine execution skill. The harness
therefore gains execution-specific gates (§Gates) including a **causality
probe** that factors never needed.

## The genome: `ExecutionProgram`

One evolution unit = one Python class implementing a shared contract
(mirroring `BaseFactor` exactly — same validator pattern, same in-memory
compilation, same persistence path):

```python
@register_executor
class VolTargetedThreshold(BaseExecutor):
    executor_id = "vol_targeted_threshold"
    regime = "per_underlying"          # or "cross_sectional" — which book shape it builds
    inputs = ["signal", "close", "volume"]   # panel/state fields it reads
    params = {"entry_z": 1.0, "exit_z": 0.5, "vol_halflife": 20}  # jitterable

    def step(self, t, signal_row, state_row, book) -> pd.Series:
        """One bar: signal + causal state + CURRENT BOOK → target weights row.

        `book` carries current positions and per-name unrealised P&L, so
        path-dependent logic (stops, profit-taking, position-aware
        rebalancing) is first-class.  The canonical, always-available API.
        """

    def target_weights(self, signal, state) -> pd.DataFrame:
        """OPTIONAL vectorised fast-path for path-INdependent programs:
        (T × N) signal + state → (T × N) weights in one shot.  The harness
        uses it when implemented (the seeds are path-independent); otherwise
        it drives `step` bar-by-bar."""
```

* **Inputs.** `signal` is the strategy's composite signal (the fitted model's
  prediction — the executor never sees the raw factors, so alpha and execution
  stay separable); `state` is a dict of **causal** context frames computed by
  the harness, never by the program (so input causality is guaranteed by
  construction): trailing volatility, ADV/liquidity, spread (where the feed
  carries it), running strategy drawdown, bar-age of the current signal.
  ``book`` (the stepwise path) additionally exposes the executor's *own*
  current positions and per-name unrealised P&L — locked decision 7.
* **Stepwise vs vectorised.** Path-dependence means weights at *t* can depend
  on the executor's own past outputs, so the canonical evaluation is a
  bar-by-bar loop (cheap on daily SP100: ~2.5k bars; the vectorised fast-path
  keeps intraday feasible later). The causality probe (§Gates) is unaffected —
  it compares outputs at *t* under future-only perturbations either way.
* **Output.** Target weights per (bar, ticker), which the existing execution/
  cost layer turns into trades. Hard output contracts (validated, not
  trusted): finite, within per-name and gross-leverage bounds, and
  dollar-neutral within tolerance when `regime == "cross_sectional"`.
* **Expressible ideas** (the creative space we are opening): volatility
  targeting, asymmetric entry/exit bands (hysteresis), signal-strength-scaled
  sizing, decay-aware holding tied to the factor's `prediction_horizon`,
  drawdown-triggered de-risking, turnover budgeting, liquidity-aware capping,
  time-stop exits.
* **Seeds = today's two hardcoded pipelines**, re-expressed as executor
  programs (`topk_dollar_neutral`, `zscore_threshold_equal_weight`). This
  guarantees byte-compatible continuity (the baseline arm *is* genome #0) and
  gives the search a working starting population without any LLM call.
* **Params block** (integer/float constants) is the jitter surface — the same
  window-jitter operator doubles as the plateau probe, unchanged.

## Fitness: the CORE objective vector (Pareto axes) + gates

Scored by a deterministic harness that runs the program through the
**existing spread-aware cost layer** on the development split. All axes
maximised; no scalar weights (locked decision 1 of the factor design holds).

1. **Net-of-cost OOS Sharpe** *(primary)* — the deflated, VAL-window Sharpe of
   the book the program builds from the evaluation signal(s), after
   transaction costs. (Gate form: deflated-Sharpe probability > 0.5 given the
   controller's `N_trials`.)
2. **Cross-signal generalisation** — mean net Sharpe across **K different
   composite signals** (different strategies/preruns/model types) minus
   λ·dispersion. An execution program must improve *how alphas are traded in
   general*, not co-adapt to one alpha — this is the execution analogue of the
   factor researcher's LOCO axis, and the single most important new
   overfitting defence at this layer.
3. **Cost efficiency** — net÷gross capture ratio (how much of the signal's
   gross P&L survives execution), with turnover as the diagnostic behind it.
4. **Parsimony** — `−complexity` (same AST count as factors).

**Hard gates** (all must pass, else treated as dominated):

* **Validity** — output contract violations (NaN/inf weights, leverage or
  per-name bound breaches, dollar-neutrality tolerance) = instant fail.
* **Causality probe** *(new, execution-specific)* — the program's weights at
  bar `t` must be bit-identical when everything after `t` changes. Implement
  as **truncation-replay** (re-run the program on the panel/signal physically
  truncated at `t` and compare the weight row at `t`), not value perturbation
  — this is the same mechanism the factor reward channel now uses to prove
  TEST-invariance, it reuses the window-keyed cache, and it cannot be dodged
  by perturbation-insensitive ops. Run at two probe points per evaluation.
  This mechanically catches full-sample standardisation, future-window ops
  and other look-ahead the factor validator could not check statically. Note
  it is *complementary* to the dev-slice (§Leak-free conventions): the
  dev-slice removes TEST; the probe catches look-ahead *within* the
  development window.
* **Turnover ceiling** — annualised turnover ≤ τ_turn (config; default set
  from the baseline pipelines' measured turnover × a headroom factor).
* **IS→VAL degradation** — net Sharpe ratio ≥ τ_deg with matching sign (same
  form as the factor gate).
* **Deflated-Sharpe / N_trials** — the search is a multiple-testing machine;
  every scored program bills the counter (same billing rule: only *scored*
  candidates count).
* **Min-activity floor** — programs that go flat everywhere trivially pass
  risk gates; require a minimum fraction of bars with non-zero book.

**Diagnostics (teacher channel, deterministic → NL brief):** per-signal Sharpe
table, turnover decomposition, cost drag, drawdown profile, exposure
timeline, which gate failed and by how much, plateau/jitter table, nearest
archived executor by behavioural correlation (books it builds vs theirs).

## Leak-free evaluation conventions (inherited, 2026-07 update)

Since this design was written, the factor reward channel was hardened
(uncommitted 2026-07 changes to `research_eval/harness.py` +
`mcp/research_service.py`, poison-invariance tests in
`tests/test_research_eval_harness.py` / `test_evolution_loop.py`). The
execution harness **must be born with the same conventions** — they supersede
any looser phrasing elsewhere in this doc:

1. **Dev-slice: TEST is physically absent at research time.** The evaluation
   service slices the panel (and here also the frozen signals and the causal
   state frames) to IS∪VAL *before* any candidate code runs — an executor's
   `step`/`target_weights` can never read a TEST price, spread or volume,
   accidentally or adversarially. The split masks are re-expressed inside the
   dev window; the full split sizes are still recorded for provenance
   (`heldout_test_split_sizes` pattern).
2. **No boundary label may reach past the available window.** The factor side
   drops any row whose `t+h` forward-return label leaves the window
   (`_label_available_mask`). Execution's analogue is milder but real: the
   mark-to-market return of the **last dev bar** needs the next bar's price,
   which is TEST — that bar is dropped from every candidate's P&L, uniformly
   (same spirit as the flat-book-at-fold-start convention). Likewise the last
   IS bar's return must not consume the first VAL price in any IS-side
   fitting/diagnostic.
3. **No full-sample statistics anywhere in the harness.** Any standardisation
   or regression inside scoring uses explicit stat masks (IS stats for VAL
   scores, dev stats for dev-wide diagnostics) — the harness never gives a
   candidate a full-sample z-score for free. For executors this is mostly
   moot (programs standardise internally and the causality probe polices
   them), but harness-side diagnostics (behavioural-correlation vs archive,
   exposure stats) follow the same rule.
4. **Caches are keyed by the exact row window.** The signal/state/book caches
   must include a `(len, first_ts, last_ts, columns)` window key
   (`_panel_window_key` pattern) — the same program evaluated on different
   slices (dev vs walk-forward folds vs TEST pass) must never collide.
5. **Poison-invariance tests are the acceptance criterion.** P0 ships the
   execution twin of the factor tests: corrupt every TEST row of the panel,
   signals and state → every candidate's full fitness dict must be
   bit-identical. This, not code review, is what proves gates 1–2.

## Overfitting / multiple-testing protocol

Identical seams to the factor design, reused as code, not re-implemented
(note the harness helpers now carry explicit `available_mask`/`stat_mask`
arguments — the exec harness consumes the updated signatures):

* **Three-way temporal split** — IS (fit any internal state / pick nothing),
  VAL (fitness; burned by the search), TEST (touched once by the Statistician
  on final archive survivors). `research_eval.splits.three_way_split`.
* **CPCV over IS∪VAL** for the robustness dispersion; **walk-forward** for the
  thesis results pass (`research_eval.splits` unchanged — they operate on
  masks, agnostic to what is being scored).
* **N_trials-aware deflation + PBO** — `research_eval.deflation.deflated_
  sharpe_ratio` and `pbo_cscv` over the candidate programs' per-bar P&L
  matrix (PBO finally gets its natural input here: N candidate *return
  streams*).
* **Cross-signal axis** (above) — the layer-specific defence: K evaluation
  signals are fixed per run (frozen list of strategy/prerun signals), so
  every candidate faces the identical panel of alphas. **Freeze = materialise:**
  the K signals are computed **once** at run start (models fit on IS only,
  with the label-availability discipline above, on the dev window only) and
  persisted as frames with the run checkpoint — never refit inside the loop.
  Each frozen frame gets the poison-invariance audit once at freeze time; a
  leaky evaluation signal would silently launder look-ahead into *every*
  executor score, which is the worst leak available at this layer.
* The **cost model itself is a held constant** across all candidates; a
  sensitivity re-score at ±50% cost is reported as a diagnostic so cost-model
  gaming is visible.

## Agent pipeline (one generation)

Reuses the factor researcher's evolution machinery — the controller, NSGA-II
selection, islands, dedup fingerprints, reflection-brief pattern and debate
are **genome-agnostic already**; only the genome type, the mutation prompts
and the evaluation call differ.

```
 Prompter ─▶ Hypothesis ─▶ [Debate]* ─▶ Codegen ─▶ Exec harness ─▶ Evolution
 (archive gaps +           (execution   (skeptic:   (BaseExecutor  (net-Sharpe     controller
  baseline diagnostics;    mechanism,   costs/turn- validator +    Pareto axes +   (reused)
  which regimes/ideas      turnover     over/lever- smoke + jitter causality probe,
  are unexplored)          budget,      age realism, probes)       cost-aware,
                           risk story)  redundancy)                multi-signal)
                                            ▲                            │
                          Reflection (deterministic brief) ◀─────────────┘
```

* **Hypothesis** declares the execution *mechanism* (e.g. "momentum signals
  decay in ~h bars, so holding beyond h buys pure noise — exit at 0.5·h under
  high vol"), a falsifiable **expected effect** (e.g. "reduces turnover ≥20%
  at ≤10% gross-capture loss") checked against realized diagnostics — the
  execution analogue of `expected_sign`.
* **Debate skeptic** attacks: cost realism, capacity, leverage disguises,
  redundancy with archived executors, backtest-artifact exploitation.
* **RAG/GraphRAG hookup**: the existing `knowledge/` stack retrieves execution
  and market-microstructure literature (transaction costs, optimal execution,
  volatility targeting, trend-following capacity studies); mechanism nodes
  gain an `execution` flavour. Reuse, not rebuild.

## Module layout

```
quant_fund_agent/
  execution/
    base.py              # BaseExecutor contract + register_executor + registry
    seeds.py             # today's two pipelines re-expressed as seed executors
    state.py             # causal state-frame builders (vol, ADV, drawdown, signal age)
    codegen.py           # validator (allowlist imports, contract checks) + in-memory compile (reuse factors/inmem pattern)
  research_eval/
    exec_harness.py      # evaluate_executor(program, signals[K], panel, split) → FitnessResult
                         #   (reuses splits/deflation/fitness/ObjectiveVector/GateResults as-is)
  agents/execution_research/
    evolution/           # thin: genome shim + mutation prompts; controller/loop REUSED from
                         # agents/factor_research/evolution (generalised where needed)
run_execution_evolution.py   # entrypoint; each run persists into the workspace Scope seam
```

**Deployment seam:** `StrategySpec`/`StrategyRecord` gains an `executor_id`
(exactly like `position_construction` today — stamped once, reproduced
identically by Architect-IS-fit, Statistician-OOS and the walk-forward trade
loop). The default remains the baseline executor, so nothing changes for
existing runs; an evolved executor is opt-in per strategy or per run
(`run_fund.py --executor <id>` / `QF_EXECUTOR`). The three divergent
signal→position implementations are unified onto `execution/` in P0 — that
consolidation is valuable even if evolution never runs.

## Configuration / mode switches

```
--engine {baseline,evolution}
--eval-signals <spec>          # K frozen composite signals (strategies/preruns) for the cross-signal axis
--generations N --population N --islands K
--debate {on,off}  --retrieval {none,rag,graphrag}
--turnover-ceiling X  --cost-sensitivity {on,off}
--split <IS/VAL/TEST>  --cpcv-folds N  --walk-forward d0,d1,…
--hypothesis-model M --debate-model M --codegen-model M   (reused)
```

## Thesis experiment matrix

| Arm | Claim under test |
| --- | --- |
| baseline pipelines (today) | control |
| + evolution (jitter-only, no LLM) | does *search itself* beat hand-tuning? |
| + LLM-semantic mutation | does LLM invention beat parameter search? |
| + debate | does critique cut overfit / raise net OOS Sharpe? |
| single-signal vs K-signal fitness | does cross-signal scoring generalise better OOS? |

Headline metrics, **OOS only**: net-of-cost deflated Sharpe, PBO over the
candidate P&L matrix, net÷gross capture, turnover, max drawdown — each
compared against the baseline executor on the identical signals.

## Phasing

* **P0 — Execution seam + deterministic harness (build first).** `execution/`
  package: `BaseExecutor`, the two seed executors (with equivalence tests
  proving byte-identical books to the current pipelines), causal state
  builders, validator; `research_eval/exec_harness.py` with all gates incl.
  the truncation-replay causality probe **and the §Leak-free conventions
  baked in from the first commit** (dev-slice, boundary-bar drop,
  window-keyed caches, poison-invariance tests — the exec twins of
  `test_research_eval_harness.py`'s leak tests); cost-aware scoring through
  the existing execution layer. No LLM. *This alone removes the
  research/deployment drift and is independently useful.*
* **P1 — Minimal loop.** Reuse the evolution controller; jitter-only mutation
  over seed params (also validates the plateau probe here); prove the search
  beats hand-tuned defaults on VAL without any LLM.
* **P2 — LLM-semantic mutation + reflection briefs** (execution prompts).
* **P3 — Debate + RAG grounding** (execution/microstructure literature).
* **P4 — Deployment integration**: `executor_id` on StrategySpec, walk-forward
  backtest consuming evolved executors, Statistician TEST pass.
* **P5 — Walk-forward final validation + the ablation matrix** for the thesis.

Documentation convention (adopted from the factor build, 2026-07): each
finished phase-group gains a **walkthrough notebook** (every component in
isolation, offline/synthetic, no API key) and a **live-run notebook** (tiny
end-to-end run with real LLM calls, showing intermediate I/O at every stage)
under `notebooks/`, referenced from `IMPLEMENTATION_PROGRESS.md`.

## Key risks

* **Backtest-artifact gaming** — the biggest one at this layer; mitigated by
  the causality probe, the cost-sensitivity re-score, turnover/leverage
  gates, and debate's "artifact exploitation" attack line.
* **Cost-model realism** — evolved programs are only as honest as the spread/
  cost model they are scored under (open question 1 below).
* **Signal co-adaptation** — mitigated by the K-signal axis; K must contain
  genuinely different alphas (open question 2).
* **Path-dependence × CPCV** — a stepwise executor's book cannot be "purged"
  mid-path the way labels can. Convention: every CPCV/walk-forward fold is
  evaluated **from a flat book at the fold's first bar** (plus the usual
  purge/embargo on the *scoring*), identically for every candidate; each
  fold's **last bar is dropped from scoring** when its mark-to-market return
  would need a price outside the fold's available window (§Leak-free
  conventions, point 2). This slightly penalises slow-entering programs
  uniformly — a fair, documented bias, not a leak. State it in the harness
  docstring and thesis.
* **Compute** — portfolio backtests are cheap relative to LOCO refits (a
  bar-loop over ~2.5k daily bars is trivial); the panel/signal caches from
  the factor build carry over. The vectorised fast-path exists for the
  intraday arm later.

## Resolved questions (author decisions, 2026-07-03)

1. **Cost model** → evolve against the current spread-aware layer; ±50%
   sensitivity re-score as diagnostic. Impact model = later orthogonal upgrade.
2. **Evaluation signals** → 3–5 diverse frozen signals from existing
   strategies/preruns (different model families × different factor subsets).
3. **Universe** → daily SP100 first (thesis consistency with the factor work);
   LOBSTER intraday reserved as a later richness arm. *(Default accepted.)*
4. **Turnover ceiling** → measured baseline-executor turnover × 3 headroom,
   config-overridable. *(Default accepted.)*
5. **State scope** → core set {vol, ADV, spread, drawdown, signal age}
   **plus book state** (positions + unrealised P&L) → stepwise contract.
   Market-regime flags deferred.
6. **LLM regime overlay** → **dropped entirely**; runtime is fully
   deterministic, no ablation arm for runtime LLM agency.

## References

- DeepMind — *FunSearch* (Nature 2024); *AlphaEvolve* (2025).
- Bailey & López de Prado — *Deflated Sharpe* (2014); *PBO* (2016).
- Almgren & Chriss — *Optimal Execution of Portfolio Transactions* (2000).
- Gârleanu & Pedersen — *Dynamic Trading with Predictable Returns and
  Transaction Costs* (2013).
- Harvey et al. — *…and the Cross-Section of Expected Returns* (2016).
- Hu et al. — *ADAS: Automated Design of Agentic Systems* (2024) — surveyed,
  set aside.
- Yu et al. — *FinMem / FinAgent* (2023–24) — surveyed, ablation-only.
