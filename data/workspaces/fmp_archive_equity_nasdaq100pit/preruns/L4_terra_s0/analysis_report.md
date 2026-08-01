# L4 (evolution, graphrag + 8 mechanism groups, GPT-5.6 Terra, seed 0) — exhaustive run analysis

Generated 2026-07-30 from run artifacts (lineage, gen_quality, prequential, state, factor DB, usage).
TEST tail untouched — all numbers are dev-window or honest prequential OOS.

```
=== 1. RUN OVERVIEW ===
trials: 798 | kept_pool: 798 | final archive: 44
usage by role:
  brainstorm   calls=   1 in= 0.35M out= 0.07M $   1.92 errors=0
  codegen      calls=   1 in= 0.64M out= 0.34M $   6.75 errors=0
  crossover    calls= 343 in= 5.34M out= 1.19M $  31.20 errors=62
  mutation     calls= 529 in= 7.56M out= 1.86M $  46.83 errors=94
  TOTAL        calls=874 $86.70 errors=156

=== 2. OPERATOR MIX (billed candidates) ===
  llm_semantic           n= 354 selectable= 354 mean_marginal=-0.0024 max=+0.0136
  crossover              n= 188 selectable= 188 mean_marginal=-0.0023 max=+0.0121
  seed                   n=  76 selectable=  76 mean_marginal=-0.0018 max=+0.0212
  cross_group            n=  74 selectable=  74 mean_marginal=-0.0012 max=+0.0205
  jitter                 n=  73 selectable=  73 mean_marginal=-0.0021 max=+0.0105
  llm_semantic_creative  n=  33 selectable=  33 mean_marginal=-0.0026 max=+0.0094

=== 3. GENERATION TRAJECTORY ===
  gen  billed  archive kept_pool mean_mv   max_mv   novelty evict
    0     76      40       76 +0.0014  +0.0212  0.850   16
    1     43      47      119 -0.0007  +0.0253  0.825   25
    2     36      71      155 -0.0008  +0.0253  0.802    3
    3     46      76      201 -0.0000  +0.0221  0.781   20
    4     39      95      240 -0.0000  +0.0221  0.772    2
    5     41      79      281 -0.0007  +0.0092  0.762   37
    6     43      93      324 -0.0005  +0.0112  0.764    3
    7     42      81      366 +0.0007  +0.0181  0.760   32
    8     41      95      407 +0.0006  +0.0181  0.767    1
    9     39      84      446 -0.0017  +0.0111  0.790   21
   10     45     101      491 -0.0012  +0.0133  0.782    2
   11     38      71      529 +0.0004  +0.0165  0.790   41
   12     37      82      566 -0.0001  +0.0165  0.810    3
   13     33      54      599 +0.0011  +0.0117  0.819   39
   14     34      58      633 +0.0012  +0.0117  0.816    1
   15     40      52      673 +0.0008  +0.0246  0.822   19
   16     38      62      711 +0.0013  +0.0246  0.823    0
   17     41      59      752 +0.0006  +0.0307  0.833   20
   18     37      62      789 +0.0009  +0.0307  0.836    8
   19      5      42      794 +0.0020  +0.0357  0.837   22
   20      4      44      798 +0.0019  +0.0357  0.839    0

=== 4. PREQUENTIAL (honest OOS on never-seen blocks) ===
  idx gen  window                        combined_OOS_IC   PBO    n_obs archive
    0    1  2015-04-01 -> 2015-11-18   +0.1207      0.35714285714285715    15757   40
    1    3  2015-11-18 -> 2016-07-12   +0.0863      0.5428571428571428    16092   71
    2    5  2016-07-12 -> 2017-03-02   +0.1023      0.3    16221   95
    3    7  2017-03-02 -> 2017-10-19   +0.0758      0.7428571428571429    16331   93
    4    9  2017-10-19 -> 2018-06-12   +0.0187      0.42857142857142855    16258   95
    5   11  2018-06-12 -> 2019-02-01   +0.0044      0.4714285714285714    16272   101
    6   13  2019-02-01 -> 2019-09-23   +0.0508      0.45714285714285713    16567   82
    7   15  2019-09-23 -> 2020-05-13   -0.0053      0.22857142857142856    16659   58
    8   17  2020-05-13 -> 2020-12-31   +0.0553      0.5    16671   62
    9   19  2020-12-31 -> 2021-08-26   +0.0954      0.3142857142857143    16880   62
  mean=+0.0604 median=+0.0656 min=-0.0053 max=+0.1207 positive=9/10

=== 5. FINAL ARCHIVE (the 44-factor book) ===
  factor_id                                          grp gen op            marginal indep   pars novelty
  stale_surprise_fear_state_cascade_diffusion        0  16 cross_group  +0.0357 +0.0000  -317 0.846
  surprise_rupture_opposing_flow_release             3   3 cross_group  +0.0276 +0.0000  -188 0.944
  earnings_downside_auction_persistence_gate         5  17 crossover    +0.0113 +0.0000  -180 0.859
  earnings_turnover_impedance_resolution             7  12 llm_semantic_creative +0.0109 +0.0000   -97 0.826
  endogenous_sell_cluster_relaxation                 5  13 llm_semantic +0.0065 +0.0000   -70 0.754
  consumption_tail_semibeta_buffered_derisking       6  17 llm_semantic +0.0064 +0.0000   -71 0.793
  participation_weighted_downside_kernel_release     5  17 llm_semantic +0.0061 +0.0000   -49 0.796
  peer_hawkes_contagion_underreaction                2   3 llm_semantic +0.0045 +0.0000  -106 0.793
  coherent_surprise_response_gap_drift               0  17 llm_semantic +0.0044 +0.0000   -66 0.848
  earnings_disambiguated_sector_liquidity_repair_j   5  20 jitter       +0.0042 +0.0000  -312 0.889
  peer_cashcycle_muted_reaction_fragility            1  17 crossover    +0.0041 +0.0000  -115 0.827
  idio_shock_relaxation_replenishment_phase          5   9 llm_semantic +0.0041 +0.0000   -85 0.875
  persistent_absorption_range_compression_j          3  19 jitter       +0.0039 +0.0000   -89 0.755
  consumption_tail_amplification_premium_h6          6  17 llm_semantic +0.0038 +0.0000  -103 0.813
  funding_gated_absorbed_flow_polarity               1  17 cross_group  +0.0035 +0.0000  -148 0.832
  consensus_revision_unassimilated_gap               1   1 llm_semantic +0.0033 +0.0000   -42 0.857
  leave_one_out_industry_impulse_lag                 4  17 llm_semantic +0.0029 +0.0000   -81 0.869
  recovery_bounce_sell_schedule_continuation         7  12 llm_semantic +0.0028 +0.0000   -75 0.801
  buyback_financed_eps_quality_unwind                1   0 seed         +0.0026 +0.0000   -68 0.913
  reported_buyback_execution_clock                   7   9 llm_semantic_creative +0.0024 +0.0000   -57 0.794
  downside_range_innovation_absorption_release       5  12 llm_semantic +0.0023 +0.0000   -78 0.804
  news_synchrony_forcedflow_conversion_j_1           5  19 jitter       +0.0022 +0.0000  -249 0.883
  broad_peer_flow_absorption_repair                  2  12 llm_semantic +0.0019 +0.0000   -77 0.847
  tail_cluster_resilient_liquidity_release           6  15 cross_group  +0.0010 +0.0000  -142 0.866
  forecast_operating_leverage_translation            0  18 llm_semantic +0.0005 +0.0000   -48 0.785
  downside_innovation_cluster_continuation           5   6 llm_semantic -0.0002 +0.0000   -60 0.813
  working_capital_funding_deterioration_drift        1   6 llm_semantic -0.0013 +0.0000   -48 0.894
  idiosyncratic_downside_hawkes_branch_pressure      7   1 llm_semantic -0.0016 +0.0000   -67 0.856
  coherent_revision_ratchet_drift                    0   8 llm_semantic -0.0020 +0.0000   -62 0.880
  industry_delayed_information_elasticity            4  16 llm_semantic -0.0020 +0.0000   -94 0.892
  uniform_sell_participation_backlog                 7  12 llm_semantic -0.0020 +0.0000   -50 0.754
  synchrony_gated_quality_defense_j_j                2  20 jitter       -0.0030 +0.0000  -186 0.851
  disclosed_float_overhang_capacity                  3   7 llm_semantic_creative -0.0036 +0.0000   -74 0.852
  sparse_liquidity_vacuum_repair_h6                  3  15 llm_semantic -0.0038 +0.0000   -46 0.754
  conditional_impact_copula_information_residual     3  13 llm_semantic_creative -0.0047 +0.0000   -63 0.763
  funding_fragility_idio_semivariance_innovation     5  15 llm_semantic -0.0051 +0.0000   -64 0.821
  volume_marked_excitation_directional_pressure_j    3  18 jitter       -0.0057 +0.0000  -160 0.907
  cashfunded_net_payout_revision_capacity            0  16 llm_semantic -0.0067 +0.0000   -90 0.888
  downside_innovation_earnings_cascade_drift         5   3 llm_semantic -0.0068 +0.0000   -72 0.847
  consensus_margin_revision_wedge                    0  11 llm_semantic -0.0080 +0.0000   -28 0.785
  quality_update_relative_price_assimilation         6   2 llm_semantic -0.0083 +0.0000   -56 0.900
  nonlinear_downside_coskew_compensation_h6          6  12 llm_semantic -0.0096 +0.0000   -54 0.812
  fear_onset_financing_state_innovation_relay        4  15 llm_semantic +0.0000 +0.0000  -114 0.899
  consensus_revision_whipsaw_correction              4  18 llm_semantic +0.0000 +0.0000   -48 0.870
  book age distribution (generation born): {0: 1, 1: 2, 2: 1, 3: 3, 6: 2, 7: 1, 8: 1, 9: 2, 11: 1, 12: 6, 13: 2, 15: 4, 16: 3, 17: 8, 18: 3, 19: 2, 20: 2}
  book mechanism-group distribution: {0: 6, 1: 5, 2: 3, 3: 6, 4: 4, 5: 10, 6: 5, 7: 5}
  categories: {'microstructure': 20, 'sentiment': 9, 'other': 5, 'volatility': 5, 'statistical_arbitrage': 3, 'mean_reversion': 2}
```

```
dev window: 2010-01-04 -> 2021-08-26 (2932 bars); TEST tail untouched
IS bars=2610 VAL bars=322

=== COMBINED MODEL (fit IS, scored VAL) ===
LightGBM combined VAL IC (44 factors): +0.1035
Ridge    combined VAL IC (44 factors): +0.0436

=== PER-FACTOR: solo VAL IC | LOCO marginal on final book ===
  stale_surprise_fear_state_cascade_diffusion          solo=+0.0277  LOCO_marginal=+0.0325
  surprise_rupture_opposing_flow_release               solo=+0.0213  LOCO_marginal=+0.0226
  working_capital_funding_deterioration_drift          solo=-0.0052  LOCO_marginal=+0.0100
  participation_weighted_downside_kernel_release       solo=-0.0233  LOCO_marginal=+0.0071
  peer_hawkes_contagion_underreaction                  solo=+0.0792  LOCO_marginal=+0.0068
  endogenous_sell_cluster_relaxation                   solo=+0.0004  LOCO_marginal=+0.0064
  disclosed_float_overhang_capacity                    solo=-0.0073  LOCO_marginal=+0.0063
  consensus_margin_revision_wedge                      solo=-0.0196  LOCO_marginal=+0.0061
  leave_one_out_industry_impulse_lag                   solo=+0.0246  LOCO_marginal=+0.0053
  conditional_impact_copula_information_residual       solo=-0.0137  LOCO_marginal=+0.0052
  peer_cashcycle_muted_reaction_fragility              solo=-0.0156  LOCO_marginal=+0.0050
  persistent_absorption_range_compression_j            solo=-0.0028  LOCO_marginal=+0.0041
  funding_gated_absorbed_flow_polarity                 solo=-0.0050  LOCO_marginal=+0.0040
  sparse_liquidity_vacuum_repair_h6                    solo=-0.0020  LOCO_marginal=+0.0039
  consumption_tail_semibeta_buffered_derisking         solo=+0.0152  LOCO_marginal=+0.0038
  reported_buyback_execution_clock                     solo=+0.0087  LOCO_marginal=+0.0035
  cashfunded_net_payout_revision_capacity              solo=+0.0009  LOCO_marginal=+0.0035
  volume_marked_excitation_directional_pressure_j      solo=-0.0583  LOCO_marginal=+0.0034
  idiosyncratic_downside_hawkes_branch_pressure        solo=+0.0218  LOCO_marginal=+0.0031
  funding_fragility_idio_semivariance_innovation       solo=+0.0078  LOCO_marginal=+0.0031
  downside_innovation_earnings_cascade_drift           solo=+0.0019  LOCO_marginal=+0.0027
  earnings_downside_auction_persistence_gate           solo=-0.0176  LOCO_marginal=+0.0023
  earnings_turnover_impedance_resolution               solo=+0.0066  LOCO_marginal=+0.0021
  consensus_revision_unassimilated_gap                 solo=-0.0111  LOCO_marginal=+0.0019
  uniform_sell_participation_backlog                   solo=+0.0239  LOCO_marginal=+0.0019
  synchrony_gated_quality_defense_j_j                  solo=-0.0152  LOCO_marginal=+0.0015
  downside_range_innovation_absorption_release         solo=-0.0109  LOCO_marginal=+0.0014
  earnings_disambiguated_sector_liquidity_repair_j     solo=+0.0065  LOCO_marginal=-0.0005
  idio_shock_relaxation_replenishment_phase            solo=+0.0187  LOCO_marginal=-0.0005
  downside_innovation_cluster_continuation             solo=+0.0315  LOCO_marginal=-0.0007
  nonlinear_downside_coskew_compensation_h6            solo=-0.0279  LOCO_marginal=-0.0009
  broad_peer_flow_absorption_repair                    solo=+0.0032  LOCO_marginal=-0.0011
  recovery_bounce_sell_schedule_continuation           solo=-0.0006  LOCO_marginal=-0.0013
  news_synchrony_forcedflow_conversion_j_1             solo=-0.0147  LOCO_marginal=-0.0013
  coherent_surprise_response_gap_drift                 solo=+0.0032  LOCO_marginal=-0.0014
  consumption_tail_amplification_premium_h6            solo=+0.0235  LOCO_marginal=-0.0015
  quality_update_relative_price_assimilation           solo=-0.0014  LOCO_marginal=-0.0018
  buyback_financed_eps_quality_unwind                  solo=-0.0022  LOCO_marginal=-0.0022
  forecast_operating_leverage_translation              solo=+0.0067  LOCO_marginal=-0.0026
  tail_cluster_resilient_liquidity_release             solo=+0.0158  LOCO_marginal=-0.0084
  coherent_revision_ratchet_drift                      solo=-0.0000  LOCO_marginal=+0.0000
  fear_onset_financing_state_innovation_relay          solo=  n/a  LOCO_marginal=+0.0000
  industry_delayed_information_elasticity              solo=  n/a  LOCO_marginal=+0.0000
  consensus_revision_whipsaw_correction                solo=  n/a  LOCO_marginal=+0.0000

=== BOOK DIVERSITY (dev window) ===
mean |pairwise corr| = 0.078 | median = 0.036 | max = 0.617
most correlated pair: sparse_liquidity_vacuum_repair_h6 ~ downside_range_innovation_absorption_rel (rho=+0.617)
participation ratio (effective independent factors): 24.0 / 44

=== ARCHIVE DIAGNOSTICS (recorded by the run) ===
  factor_id                                            IS_IC   degrad  max|corr| coverage
  coherent_revision_ratchet_drift                      +0.000    None  0.000   1.00
  consensus_margin_revision_wedge                      -0.001    None  0.535   1.00
  cashfunded_net_payout_revision_capacity              -0.005  -0.05  0.460   1.00
  stale_surprise_fear_state_cascade_diffusion          +0.005    None  0.033   1.00
  coherent_surprise_response_gap_drift                 -0.013  -0.29  0.421   1.00
  forecast_operating_leverage_translation              -0.010  -0.66  0.491   1.00
  buyback_financed_eps_quality_unwind                  -0.001    None  0.560   1.00
  consensus_revision_unassimilated_gap                 +0.008  -1.79  0.378   1.00
  working_capital_funding_deterioration_drift          -0.005    None  0.528   1.00
  peer_cashcycle_muted_reaction_fragility              -0.011  +1.47  0.528   1.00
  funding_gated_absorbed_flow_polarity                 -0.002    None  0.165   1.00
  peer_hawkes_contagion_underreaction                  +0.031  +2.50  0.455   1.00
  broad_peer_flow_absorption_repair                    +0.008  -0.48  0.243   1.00
  synchrony_gated_quality_defense_j_j                  +0.002    None  0.099   1.00
  surprise_rupture_opposing_flow_release               +0.005    None  0.246   1.00
  disclosed_float_overhang_capacity                    -0.001    None  0.359   1.00
  conditional_impact_copula_information_residual       -0.017  +0.85  0.483   1.00
  sparse_liquidity_vacuum_repair_h6                    +0.012  -0.17  0.615   1.00
  volume_marked_excitation_directional_pressure_j      -0.011  +5.23  0.455   1.00
  persistent_absorption_range_compression_j            -0.001    None  0.067   1.00
  fear_onset_financing_state_innovation_relay          +0.000    None  0.000   1.00
  industry_delayed_information_elasticity              +0.000    None  0.000   1.00
  leave_one_out_industry_impulse_lag                   +0.009  +2.63  0.245   1.00
  consensus_revision_whipsaw_correction                +0.000    None  0.000   1.00
  downside_innovation_earnings_cascade_drift           +0.005  +0.76  0.612   1.00
  downside_innovation_cluster_continuation             +0.014  +2.19  0.484   1.00
  idio_shock_relaxation_replenishment_phase            +0.018  +0.91  0.290   1.00
  downside_range_innovation_absorption_release         +0.003    None  0.655   1.00
  endogenous_sell_cluster_relaxation                   +0.002    None  0.387   1.00
  funding_fragility_idio_semivariance_innovation       +0.009  +0.88  0.356   1.00
  earnings_downside_auction_persistence_gate           +0.002    None  0.612   1.00
  participation_weighted_downside_kernel_release       -0.010  +2.48  0.468   1.00
  news_synchrony_forcedflow_conversion_j_1             +0.001    None  0.154   1.00
  earnings_disambiguated_sector_liquidity_repair_j     +0.004    None  0.205   1.00
  quality_update_relative_price_assimilation           -0.001    None  0.136   1.00
  nonlinear_downside_coskew_compensation_h6            +0.008  -3.78  0.416   1.00
  tail_cluster_resilient_liquidity_release             +0.026  +0.62  0.254   1.00
  consumption_tail_semibeta_buffered_derisking         -0.005    None  0.255   1.00
  consumption_tail_amplification_premium_h6            +0.013  +1.72  0.416   1.00
  idiosyncratic_downside_hawkes_branch_pressure        +0.016  +1.40  0.543   1.00
  reported_buyback_execution_clock                     -0.022  -0.33  0.359   1.00
  earnings_turnover_impedance_resolution               +0.021  +0.28  0.488   1.00
  recovery_bounce_sell_schedule_continuation           -0.003    None  0.172   1.00
  uniform_sell_participation_backlog                   +0.007  +3.67  0.384   1.00
```

## Key findings

1. **Honest OOS is real but FRONT-LOADED, unlike L2's ramp.** The prequential probe was
   positive on 9/10 never-seen blocks, mean IC +0.060 (L2: +0.054, 10/10), but the
   trajectory is the mirror of L2's: strongest at the START (+0.121 / +0.086 / +0.102 on
   the 2015-17 reveals — the 96 graph-grounded seeds already carried real OOS edge),
   sagging mid-run (2017-19: +0.019 / +0.004 / -0.005 — the only negative block is the
   2019-09→2020-05 COVID-crash window), then recovering (+0.055 / +0.095). L2 (ungrounded,
   Opus) had to EVOLVE its edge (+0.010 early → +0.082-0.097 late); L4's retrieval
   grounding delivered it at seeding time.
2. **Combined nonlinear model doubles L2:** LightGBM VAL IC **+0.1035** (L2: +0.0430) on
   the final IS-fit/VAL-score split; Ridge only +0.0436 — over half the book's combined
   value is nonlinear/interaction structure, which is exactly what the lightgbm marginal
   combiner selects for. Top LOCO contributors are the two cross_group children
   stale_surprise_fear_state_cascade_diffusion (+0.033) and
   surprise_rupture_opposing_flow_release (+0.023) — the explicit cross-mechanism
   synthesis operator produced the book's two most valuable members.
3. **Final book: 44 factors spanning ALL 8 mechanism groups** (3-10 each; group 5
   "downside-excitation/absorption" richest at 10). Categories: 20 microstructure,
   9 sentiment/revision, 5 volatility, 5 other, 3 stat-arb, 2 mean-reversion. Age-diverse:
   survivors from gen 0-20, mode at gen 17. Mean |pairwise corr| 0.078, participation
   ratio 24.0/44 effective independent factors (L2: 13.5/18).
4. **The reveal-cull ratchet cut deeper than L2.** 315 evictions (L2: 132) in waves at
   every reveal (gens 5/7/11/13: 37/32/41/39 evictions); archive 101→42 over gens 10-19
   while max marginal ROSE to +0.0357 — the culls concentrated quality.
5. **Operator economics differ from L2:** seeds were the single best source of marginal
   value (max +0.0212, mean -0.0018 over 76 billed) and cross_group (74 children, max
   +0.0205) out-performed plain crossover (188, max +0.0121); llm_semantic max +0.0136.
   In L2 llm_semantic dominated (max +0.036) — with graph grounding, VALUE ENTERS THROUGH
   THE SEEDS and recombination, not through unguided semantic mutation.
6. **Cost/reliability: $86.70 total** (~$0.11 per scored candidate, 5× cheaper than
   L2's $0.55; Terra $2.5/$15 vs Opus $5/$25 and ~4× fewer output tokens/call).
   156/874 calls errored (18%, all in mutation/crossover roles, retried successfully —
   0 errors in the first ~500 calls, the burst came in late generations).
7. **CAVEAT — independence axis is None post-rescore.** `rescore_archive` does not
   recompute the residual-IC independence axis (it needs the LOCO book context), so every
   archive member's `independence` reads None after the last reveal (billed-time values
   were real: 445/798 nonzero). Display-only here, but crowding-distance on a None axis
   during post-reveal pruning deserves a look before the ladder is written up.
8. **CAVEAT — three degenerate (constant-signal) members:**
   fear_onset_financing_state_innovation_relay, industry_delayed_information_elasticity,
   consensus_revision_whipsaw_correction produce zero-variance signals (val_ic None,
   coverage 1.0), admitted at marginal exactly 0.0 and kept on novelty/parsimony axes.
   Harmless to the combined model (LOCO 0) but the smoke/non-degeneracy filter should
   reject constant signals; fix before the remaining arms.

## Figures

Figure suite (PDF + 300-dpi PNG) in `figures/` next to this report;
generator: `scripts/plot_l2_run_figures.py` (pointed at this run dir).
