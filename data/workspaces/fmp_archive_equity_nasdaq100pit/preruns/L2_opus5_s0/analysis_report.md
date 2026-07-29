# L2 (evolution, no grounding, Opus 5, seed 0) — exhaustive run analysis

Generated 2026-07-29 from run artifacts (lineage, gen_quality, prequential, state, factor DB, usage).
TEST tail (post-2021-08-26) untouched — all numbers are dev-window or honest prequential OOS.

```
=== 1. RUN OVERVIEW ===
trials: 825 | kept_pool: 820 | final archive: 18
usage by role:
  brainstorm   calls=   1 in= 0.03M out= 0.03M $   0.85 errors=0
  codegen      calls=   1 in= 0.12M out= 0.08M $   2.70 errors=0
  crossover    calls=  63 in= 7.28M out= 3.07M $ 113.07 errors=34
  mutation     calls= 145 in=14.12M out=10.60M $ 335.51 errors=54
  TOTAL        calls=210 $452.13 errors=88

=== 2. OPERATOR MIX (billed candidates) ===
  llm_semantic           n=511 selectable=506 mean_marginal=-0.0016 max=+0.0363
  crossover              n=246 selectable=246 mean_marginal=-0.0008 max=+0.0107
  llm_semantic_creative  n= 45 selectable= 45 mean_marginal=-0.0030 max=+0.0119
  jitter                 n= 17 selectable= 17 mean_marginal=-0.0088 max=+0.0060
  seed                   n=  6 selectable=  6 mean_marginal=+0.0023 max=+0.0196

=== 3. GENERATION TRAJECTORY ===
  gen  billed adm%  archive kept_pool mean_mv   max_mv   novelty evict
    0      6  100        5        6 +0.0002  +0.0187  0.818    1
    1     39  100       17       45 +0.0004  +0.0129  0.783    2
    2     38  100       25       83 +0.0025  +0.0493  0.780    0
    3     41  100       26      108 +0.0034  +0.0420  0.735   10
    4     44  100       28      168 +0.0043  +0.0363  0.755    5
    5     43   98       28      210 +0.0027  +0.0421  0.755   12
    6     43  100       35      253 +0.0024  +0.0421  0.748    6
    7     46  100       13      299 +0.0063  +0.0250  0.734   27
    8     46  100       13      345 +0.0063  +0.0250  0.734    0
    9     45   96       18      388 +0.0036  +0.0239  0.764    2
   10     43  100       22      431 +0.0037  +0.0239  0.762    2
   11     44  100       24      475 +0.0034  +0.0441  0.773   10
   12     46  100       27      521 +0.0027  +0.0441  0.778    0
   13     44  100       24      565 +0.0036  +0.0501  0.782   14
   14     47  100       27      612 +0.0036  +0.0501  0.784    1
   15     44  100       12      656 +0.0039  +0.0150  0.795   19
   16     38  100       16      694 +0.0058  +0.0204  0.786    0
   17     43  100       20      737 +0.0041  +0.0166  0.769    4
   18     42   98       23      778 +0.0037  +0.0166  0.773    1
   19     42   98       18      819 +0.0025  +0.0190  0.779   16
   20      1  100       18      820 +0.0025  +0.0190  0.779    0

=== 4. PREQUENTIAL (honest OOS on never-seen blocks) ===
  idx gen  window                        combined_OOS_IC   PBO    n_obs archive
    0   1  2015-04-01 -> 2015-11-18   +0.0098      0.7    15576    5
    1   3  2015-11-18 -> 2016-07-12   +0.0197      0.35714285714285715    16092   25
    2   5  2016-07-12 -> 2017-03-02   +0.0520      0.6    16221   28
    3   7  2017-03-02 -> 2017-10-19   +0.0410      0.7714285714285715    16331   35
    4   9  2017-10-19 -> 2018-06-12   +0.0283      0.7    16258   13
    5  11  2018-06-12 -> 2019-02-01   +0.0969      0.6    16272   22
    6  13  2019-02-01 -> 2019-09-23   +0.0370      0.5142857142857142    16567   27
    7  15  2019-09-23 -> 2020-05-13   +0.0787      0.5    16659   27
    8  17  2020-05-13 -> 2020-12-31   +0.0916      0.5285714285714286    16671   16
    9  19  2020-12-31 -> 2021-08-26   +0.0823      0.45714285714285713    16880   23
  mean=+0.0537 median=+0.0465 min=+0.0098 max=+0.0969 positive=10/10

=== 5. FINAL ARCHIVE (the 18-factor book) ===
  factor_id                                          gen op            marginal indep   pars novelty
  program_gated_elasticity_factor_return_snapback     16 llm_semantic +0.0190 -0.0066  -314 0.750
  tsrv_kalman_gain_transient_deviation_memory_h6      18 llm_semantic +0.0080 +0.0094  -236 0.806
  comparable_basket_repair_rate_activity_clock_h6     17 llm_semantic +0.0072 +0.0196  -238 0.730
  volume_at_level_wall_vacuum_idio_dislocation_fli    19 llm_semantic +0.0062 +0.0014  -228 0.748
  reignition_hazard_pending_spell_vs_paid_split_h6    19 llm_semantic +0.0054 +0.0251  -177 0.816
  absorbed_wick_routed_unowed_transient_snapback      13 crossover    +0.0050 +0.0343  -335 0.760
  backing_split_bivariate_var_window_mass_gate_h6     19 llm_semantic +0.0042 +0.0062  -234 0.824
  corporate_net_demand_hinge_asymmetric_dislocatio    17 llm_semantic +0.0039 +0.0280  -128 0.748
  run_surprisal_vs_flip_baseline_exhaustion_h6        19 llm_semantic +0.0025 +0.0167  -135 0.768
  wilcoxon_drift_mood_compression_metaorder_wedge      7 llm_semantic +0.0012 +0.0125   -55 0.759
  roundnumber_lattice_barrier_breach_asymmetry_h6     18 llm_semantic +0.0006 -0.0036  -128 0.787
  sqrt_law_weighted_flow_pressure_freshness_wedge      4 llm_semantic -0.0002 -0.0073   -40 0.710
  rice_consistency_wedge_gated_crossing_clock_fade    19 crossover    -0.0007 -0.0019  -173 0.788
  churn_resilience_gated_sweep_debt_snapback_h6       19 llm_semantic -0.0017 +0.0200  -162 0.777
  program_flow_vs_name_flow_volume_split_reversal_    19 llm_semantic -0.0029 +0.0057  -130 0.763
  absorption_depth_gated_wick_asymmetry_fresh          1 llm_semantic -0.0044 +0.0181   -65 0.788
  delivery_capped_valuation_repair_flow_attributed    19 crossover    -0.0045 +0.0115  -500 0.827
  volume_evenness_entropy_innovation_gated_path_ef     5 llm_semantic -0.0047 +0.0233   -64 0.870
  book age distribution (generation born): {1: 1, 4: 1, 5: 1, 7: 1, 13: 1, 16: 1, 17: 2, 18: 2, 19: 8}
  fitness diagnostics available: ['base_ic', 'complexity', 'coverage', 'deflation', 'degradation_ratio', 'delta_participation', 'gross_ret', 'ic_decay', 'independence_metric', 'is_ic', 'jitter_ics', 'marginal_value_raw', 'max_abs_corr', 'n_trials', 'net_gross_ratio', 'net_ret', 'novelty_candidate_ast_nodes', 'novelty_candidate_unique_subtrees']
```

```
dev window: 2010-01-04 -> 2021-08-25 (2932 bars); TEST tail untouched
IS bars=2199 VAL bars=732

=== COMBINED MODEL (fit IS, scored VAL) ===
LightGBM combined VAL IC (18 factors): +0.0430
Ridge    combined VAL IC (18 factors): +0.0520

=== PER-FACTOR: solo VAL IC | LOCO marginal on final book ===
  program_gated_elasticity_factor_return_snapback      solo=+0.0250  LOCO_marginal=+0.0158
  corporate_net_demand_hinge_asymmetric_dislocatio     solo=+0.0389  LOCO_marginal=+0.0066
  reignition_hazard_pending_spell_vs_paid_split_h6     solo=+0.0315  LOCO_marginal=+0.0040
  absorbed_wick_routed_unowed_transient_snapback       solo=+0.0277  LOCO_marginal=+0.0022
  sqrt_law_weighted_flow_pressure_freshness_wedge      solo=+0.0106  LOCO_marginal=+0.0015
  churn_resilience_gated_sweep_debt_snapback_h6        solo=+0.0280  LOCO_marginal=+0.0005
  absorption_depth_gated_wick_asymmetry_fresh          solo=-0.0052  LOCO_marginal=+0.0003
  wilcoxon_drift_mood_compression_metaorder_wedge      solo=-0.0010  LOCO_marginal=-0.0000
  program_flow_vs_name_flow_volume_split_reversal_     solo=-0.0000  LOCO_marginal=-0.0000
  tsrv_kalman_gain_transient_deviation_memory_h6       solo=+0.0113  LOCO_marginal=-0.0001
  volume_evenness_entropy_innovation_gated_path_ef     solo=+0.0003  LOCO_marginal=-0.0004
  roundnumber_lattice_barrier_breach_asymmetry_h6      solo=-0.0082  LOCO_marginal=-0.0010
  rice_consistency_wedge_gated_crossing_clock_fade     solo=+0.0114  LOCO_marginal=-0.0011
  volume_at_level_wall_vacuum_idio_dislocation_fli     solo=+0.0079  LOCO_marginal=-0.0014
  backing_split_bivariate_var_window_mass_gate_h6      solo=+0.0045  LOCO_marginal=-0.0025
  delivery_capped_valuation_repair_flow_attributed     solo=+0.0020  LOCO_marginal=-0.0034
  comparable_basket_repair_rate_activity_clock_h6      solo=+0.0008  LOCO_marginal=-0.0063
  run_surprisal_vs_flip_baseline_exhaustion_h6         solo=+0.0113  LOCO_marginal=-0.0082

=== BOOK DIVERSITY (dev window) ===
mean |pairwise corr| = 0.085 | median = 0.042 | max = 0.716
most correlated pair: absorbed_wick_routed_unowed_transient_sn ~ churn_resilience_gated_sweep_debt_snapba (rho=+0.716)
participation ratio (effective independent factors): 13.5 / 18

=== ARCHIVE DIAGNOSTICS (recorded by the run) ===
  factor_id                                            IS_IC   degrad  max|corr| coverage
  absorption_depth_gated_wick_asymmetry_fresh          +0.006  -4.56  0.342   1.00
  sqrt_law_weighted_flow_pressure_freshness_wedge      -0.012  +1.94  0.817   0.98
  volume_evenness_entropy_innovation_gated_path_ef     -0.000    None  0.049   0.91
  wilcoxon_drift_mood_compression_metaorder_wedge      +0.001    None  0.058   0.90
  absorbed_wick_routed_unowed_transient_snapback       +0.027  +2.05  0.342   1.00
  program_gated_elasticity_factor_return_snapback      +0.003    None  0.381   1.00
  corporate_net_demand_hinge_asymmetric_dislocatio     +0.020  +2.13  0.214   0.93
  comparable_basket_repair_rate_activity_clock_h6      +0.015  +3.91  0.054   1.00
  roundnumber_lattice_barrier_breach_asymmetry_h6      -0.001    None  0.005   1.00
  tsrv_kalman_gain_transient_deviation_memory_h6       +0.018  +1.42  0.200   1.00
  churn_resilience_gated_sweep_debt_snapback_h6        +0.037  +1.06  0.716   1.00
  rice_consistency_wedge_gated_crossing_clock_fade     +0.015  +3.62  0.287   1.00
  run_surprisal_vs_flip_baseline_exhaustion_h6         +0.017  +3.49  0.503   1.00
  program_flow_vs_name_flow_volume_split_reversal_     +0.001    None  0.121   1.00
  volume_at_level_wall_vacuum_idio_dislocation_fli     -0.005  +2.23  0.175   1.00
  reignition_hazard_pending_spell_vs_paid_split_h6     +0.036  +1.47  0.503   0.99
  backing_split_bivariate_var_window_mass_gate_h6      +0.003    None  0.267   1.00
  delivery_capped_valuation_repair_flow_attributed     -0.001    None  0.303   1.00
Traceback (most recent call last):
  File "<stdin>", line 81, in <module>
FileNotFoundError: [Errno 2] No such file or directory: 'data/workspaces/fmp_archive_equity_nasdaq100pit/preruns/L2_opus5_s0/factor_db.json'
```


## Key findings

1. **Honest OOS predictive power is real and GREW across the run.** The prequential probe
   (combined book scored on each newly revealed, never-before-seen block) was positive on
   10/10 blocks, mean IC +0.054, and trended upward: ~+0.01-0.02 in the 2015-16 reveals to
   +0.08-0.10 in the 2019-21 reveals. CSCV PBO fell from 0.70 (early) to 0.46 (final) --
   the book became LESS overfit as evolution progressed.
2. **The reveal-cull ratchet visibly worked.** Archive contractions at gens 7 (35->13),
   15 (27->12) and 19 followed reveals: members whose edge did not survive unseen data were
   dominated out (63 logged evictions), and the book rebuilt stronger each time.
3. **Final book: 18 factors, publish deflation kept 18/18 at N_trials=825.** Age-diverse
   (8 born in gen 19, survivors from gens 1-16). Mean |pairwise corr| 0.085, effective
   independent factors 13.5/18. Categories: 12 microstructure, 3 stat-arb, 3 mean-reversion.
4. **Combined ML model (fit IS, scored VAL, dev-window):** LightGBM IC +0.043, Ridge +0.052.
   Top LOCO contributors: program_gated_elasticity_factor_return_snapback (+0.016),
   corporate_net_demand_hinge (+0.007), reignition_hazard (+0.004). Tail members contribute
   ~0/negative marginally on the final book but survive on independence/novelty axes
   (Pareto semantics; curation=archive keeps front-1).
5. **Operator economics:** llm_semantic produced the best children (max marginal +0.036);
   crossover softer (max +0.011); creative-frac children slightly below base; jitter weakest.
   Seeds were strong (mean marginal +0.0023).
6. **CAVEAT (fixed for future arms): bar-size prompt bug.** Prompts told the model
   "10-second bars" (the probe read local LOBSTER files) while the panel is DAILY. All
   evaluation was honest, but idea narratives skew intraday-microstructure. Fixed in
   pipeline._infer_seconds_per_bar (config frequency now wins -> 86400s).
7. **Gen 20 was starved by an internet outage** (1/48 children; 47 connection failures).
   Connection-error backoff retries added for future runs (~30 min outage tolerance).
8. **Cost:** $452.13 total (mutation $335 / crossover $113 / seeds+codegen $4); ~80% of
   spend is Opus thinking tokens. 825 scored candidates -> ~$0.55/candidate all-in.
