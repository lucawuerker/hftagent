# L4R_terra_s0 — exhaustive run analysis

Generated 2026-08-01 from run artifacts (lineage, gen_quality, prequential, state, factor DB, usage).
TEST tail untouched — all numbers are dev-window or honest prequential OOS.

```
=== 1. RUN OVERVIEW ===
trials: 468 | kept_pool: 464 | final archive: 62
usage by role:
  brainstorm   calls= 122 in= 1.85M out= 0.18M $   7.27 errors=0
  codegen      calls= 133 in= 1.46M out= 0.80M $  15.59 errors=0
  crossover    calls=  58 in= 0.85M out= 0.20M $   5.18 errors=0
  mutation     calls= 371 in= 4.67M out= 1.33M $  31.67 errors=0
  TOTAL        calls=684 $59.71 errors=0

=== 2. OPERATOR MIX (billed candidates) ===
  refine                 n= 263 selectable= 263 mean_marginal=-0.0010 max=+0.0146
  seed                   n= 163 selectable= 159 mean_marginal=-0.0016 max=+0.0174
  cross_group            n=  42 selectable=  42 mean_marginal=-0.0033 max=+0.0109

=== 3. GENERATION TRAJECTORY ===
  gen  billed  archive kept_pool mean_mv   max_mv   novelty evict
    0     79      37       76 +0.0011  +0.0174  0.847   12
    1     22      31       98 +0.0018  +0.0200  0.816   12
    2     23      35      121 +0.0027  +0.0200  0.828    7
    3     22      41      143 +0.0007  +0.0169  0.823    7
    4     20      48      163 +0.0009  +0.0169  0.798    2
    5     22      40      185 +0.0010  +0.0167  0.806   19
    6     20      42      205 +0.0012  +0.0167  0.808    4
    7     19      44      224 -0.0002  +0.0136  0.839    9
    8     19      47      243 -0.0001  +0.0136  0.839    1
    9     20      53      262 -0.0007  +0.0153  0.829   12
   10     18      63      280 -0.0000  +0.0153  0.822    5
   11     20      62      300 -0.0003  +0.0162  0.815   13
   12     14      69      314 -0.0002  +0.0162  0.809    2
   13     19      62      333 -0.0001  +0.0193  0.811   19
   14     21      68      354 -0.0001  +0.0193  0.807    1
   15     20      63      374 -0.0004  +0.0186  0.816   12
   16     19      67      393 -0.0003  +0.0186  0.807    0
   17     18      53      411 +0.0010  +0.0345  0.817   21
   18     18      61      429 +0.0009  +0.0345  0.815    1
   19     18      59      447 +0.0001  +0.0125  0.826   10
   20     17      62      464 +0.0002  +0.0125  0.824    0

=== 4. PREQUENTIAL (honest OOS on never-seen blocks) ===
  idx gen  window                        combined_OOS_IC   PBO    n_obs archive
    0    1  2015-04-01 -> 2015-11-18   +0.1046      0.5714285714285714    15757   37
    1    3  2015-11-18 -> 2016-07-12   +0.0625      0.7142857142857143    16092   35
    2    5  2016-07-12 -> 2017-03-02   +0.0933      0.34285714285714286    16221   48
    3    7  2017-03-02 -> 2017-10-19   +0.0924      0.5142857142857142    16331   42
    4    9  2017-10-19 -> 2018-06-12   +0.0223      0.3    16258   47
    5   11  2018-06-12 -> 2019-02-01   +0.0269      0.4142857142857143    16272   63
    6   13  2019-02-01 -> 2019-09-23   +0.0814      0.4142857142857143    16567   69
    7   15  2019-09-23 -> 2020-05-13   -0.0055      0.11428571428571428    16659   68
    8   17  2020-05-13 -> 2020-12-31   +0.0961      0.14285714285714285    16671   67
    9   19  2020-12-31 -> 2021-08-26   +0.0677      0.21428571428571427    16880   61
  mean=+0.0642 median=+0.0746 min=-0.0055 max=+0.1046 positive=9/10

=== 5. FINAL ARCHIVE (the 62-factor book) ===
  factor_id                                          grp gen op            marginal indep   pars novelty
  completed_marked_arrival_absorption_reversal_h6    3  15 refine       +0.0125 +0.0041  -126 0.821
  calendar_phase_metaorder_impact_deficit            6  19 seed         +0.0118 +0.0000  -143 0.702
  episode_coherence_fresh_revision_gap               0  19 refine       +0.0094 +0.0065  -153 0.809
  shrunk_directed_lag_liquidity_gap                  1  20 refine       +0.0059 +0.0000  -256 0.629
  factor_premium_dispersion_regime_transfer          2  14 seed         +0.0051 -0.0001  -116 0.810
  sector_shrunk_mode_follower_relay                  3  19 refine       +0.0046 +0.0119  -194 0.857
  fundamental_peer_surprise_transmission_gap         1   0 seed         +0.0040 -0.0103  -191 0.878
  downside_tail_dependence_unpaid_bridge             5  20 seed         +0.0038 +0.0000  -166 0.877
  sbc_dilution_acceleration_supply_drift             0   0 seed         +0.0037 +0.0106   -56 0.856
  prompt_cooling_thin_traversal_reversal             5  13 refine       +0.0036 -0.0060  -188 0.867
  directional_recurrence_saturation_completion       2  10 refine       +0.0035 -0.0123   -97 0.799
  downside_persistence_fragility_exhaustion          5  11 refine       +0.0034 -0.0112   -88 0.803
  scheduled_flow_impact_decay_reversal               1   4 refine       +0.0034 -0.0054   -76 0.865
  idio_thin_side_vacuum_reversal_h6                  3  18 refine       +0.0031 +0.0036   -78 0.788
  directed_laggraph_liquidity_unpaid_response        1  19 refine       +0.0031 +0.0010  -218 0.842
  intraday_supply_curve_overshoot_reversal           3   2 refine       +0.0031 +0.0127   -90 0.849
  fresh_cashflow_confirmed_surprise_drift            0   3 refine       +0.0030 +0.0015   -85 0.866
  fresh_expectation_delivery_decay                   4  18 refine       +0.0025 +0.0064   -83 0.846
  sector_hawkes_compensator_late_follower            2  10 seed         +0.0021 +0.0058  -159 0.849
  quality_regime_conditioned_issuer_distribution     4  13 cross_group  +0.0019 +0.0153  -185 0.896
  volume_clock_signed_flow_schedule_h6               6  14 refine       +0.0019 -0.0032  -154 0.744
  cashflow_deterioration_idio_downside_overshoot     5   5 refine       +0.0018 -0.0057   -90 0.876
  event_freshness_accounting_confirmed_surprise      4   5 refine       +0.0018 -0.0003   -92 0.903
  cashcycle_replenishment_rebound_completion         6   6 refine       +0.0015 -0.0014   -78 0.829
  growth_expectation_convexity_break                 4  18 seed         +0.0015 +0.0014   -56 0.888
  robust_tail_equity_supply_reversal                 6   1 refine       +0.0013 -0.0125   -49 0.753
  return_volume_recurrence_intensity_acceleration    2   5 refine       +0.0013 -0.0079  -122 0.867
  growth_expectation_fresh_delivery_break            4  19 refine       +0.0013 +0.0142   -94 0.762
  tail_survival_capacity_impedance_drift             4   2 cross_group  +0.0013 +0.0100  -121 0.883
  cashcycle_ambiguity_operating_fade                 0  12 refine       +0.0009 -0.0064  -119 0.897
  report_clock_tail_network_derisking_gate           6  10 cross_group  +0.0009 +0.0146  -227 0.838
  thin_side_sqrt_impact_impulse_h6                   3   7 refine       +0.0008 +0.0099   -76 0.807
  fractional_propagator_flow_debt_reversal           7  18 seed         +0.0007 +0.0106  -118 0.735
  latent_impact_absorption_persistence               7  14 seed         +0.0007 +0.0095  -177 0.886
  scheduled_downside_cooling_information_drift       5  19 refine       +0.0002 +0.0053  -124 0.770
  surprise_novelty_absorption_drift                  7   4 refine       +0.0002 +0.0049  -139 0.882
  asymmetric_overnight_range_wedge_rebound           5  14 refine       -0.0001 -0.0054  -122 0.618
  fresh_cashfunded_upgrade_wedge                     4   1 refine       -0.0002 -0.0136   -52 0.828
  stabilized_convex_common_shock_overshoot           1  17 refine       -0.0004 -0.0017  -170 0.786
  partial_graph_peer_innovation_catchup              4  11 seed         -0.0008 +0.0167  -153 0.876
  sector_shock_distributed_lag_unpaid_response       1  13 refine       -0.0014 +0.0136  -221 0.850
  signed_graph_curvature_residual_diffusion          2  13 seed         -0.0014 +0.0029  -123 0.798
  persistent_participation_pressure_continuation     7   3 refine       -0.0015 +0.0052   -73 0.785
  fundamental_momentum_extrapolation_fragility       4  19 seed         -0.0017 +0.0000   -91 0.901
  sector_relative_opening_auction_rejection_h6       7  19 refine       -0.0020 +0.0000  -136 0.799
  relaxation_time_orderflow_persistence_drift        3   0 seed         -0.0025 +0.0071   -60 0.785
  persistent_liquidity_commonality_derisking         3   1 refine       -0.0027 -0.0001   -48 0.713
  decelerating_execution_debt_reversal               7  20 refine       -0.0029 +0.0000   -97 0.836
  revision_financing_gated_idio_downside_polarity    5  15 cross_group  -0.0029 +0.0111  -142 0.856
  cashbacked_asymmetric_surprise_drift_h6            0   5 refine       -0.0030 +0.0110   -49 0.789
  resilient_downside_excitation_cooling_continuati   5  13 refine       -0.0036 +0.0050  -114 0.713
  cashflow_quality_confirmed_surprise_drift          0   1 refine       -0.0036 +0.0059   -46 0.789
  superlinear_downside_variance_relaxation_tilt      5   7 seed         -0.0041 -0.0046  -168 0.892
  idiosyncratic_tail_resilience_absorption_repair    6   7 refine       -0.0042 -0.0013  -224 0.845
  industry_turnover_gated_idio_repair                7   9 refine       -0.0046 +0.0141   -77 0.871
  downside_excitation_fragility_acceleration_h6      5  10 refine       -0.0056 +0.0094  -120 0.825
  cash_realization_surprise_disagreement             0   8 refine       -0.0057 -0.0010  -103 0.889
  tail_survival_forcedflow_reversal_h6               5   6 cross_group  -0.0075 -0.0049  -143 0.883
  report_assimilation_release_continuation_h6        3  17 refine       -0.0079 +0.0056  -209 0.904
  downside_reflexive_probe_persistence               5  18 refine       -0.0098 +0.0057  -116 0.813
  anticipated_report_peer_tail_impact_gate           6  12 refine       -0.0102 +0.0318  -175 0.807
  downside_pressure_freshness_h6                     5   2 refine       -0.0111 +0.0059   -55 0.813
  book age distribution (generation born): {0: 3, 1: 4, 2: 3, 3: 2, 4: 2, 5: 4, 6: 2, 7: 3, 8: 1, 9: 1, 10: 4, 11: 2, 12: 2, 13: 5, 14: 4, 15: 2, 17: 2, 18: 5, 19: 8, 20: 3}
  book mechanism-group distribution: {0: 7, 1: 6, 2: 5, 3: 8, 4: 9, 5: 13, 6: 7, 7: 7}
  categories: {'microstructure': 25, 'statistical_arbitrage': 13, 'sentiment': 10, 'volatility': 7, 'mean_reversion': 4, 'momentum': 2, 'carry': 1}
```

```
dev window: 2020-01-02 -> 2025-03-06 (1300 bars); TEST tail untouched
IS bars=1158 VAL bars=142

=== COMBINED MODEL (fit IS, scored VAL) ===
LightGBM combined VAL IC (62 factors): +0.0768
Ridge    combined VAL IC (62 factors): -0.0208

=== PER-FACTOR: solo VAL IC | LOCO marginal on final book ===
  calendar_phase_metaorder_impact_deficit              solo=+0.0162  LOCO_marginal=+0.0365
  directional_recurrence_saturation_completion         solo=-0.0035  LOCO_marginal=+0.0149
  return_volume_recurrence_intensity_acceleration      solo=+0.0034  LOCO_marginal=+0.0120
  relaxation_time_orderflow_persistence_drift          solo=-0.0083  LOCO_marginal=+0.0089
  downside_reflexive_probe_persistence                 solo=+0.0329  LOCO_marginal=+0.0071
  scheduled_flow_impact_decay_reversal                 solo=-0.0317  LOCO_marginal=+0.0060
  prompt_cooling_thin_traversal_reversal               solo=+0.0109  LOCO_marginal=+0.0034
  thin_side_sqrt_impact_impulse_h6                     solo=-0.0099  LOCO_marginal=+0.0026
  superlinear_downside_variance_relaxation_tilt        solo=-0.0096  LOCO_marginal=+0.0020
  downside_pressure_freshness_h6                       solo=-0.0269  LOCO_marginal=-0.0011
  persistent_participation_pressure_continuation       solo=-0.0493  LOCO_marginal=-0.0029
  idio_thin_side_vacuum_reversal_h6                    solo=-0.0069  LOCO_marginal=-0.0035
  persistent_liquidity_commonality_derisking           solo=-0.0258  LOCO_marginal=-0.0118
  completed_marked_arrival_absorption_reversal_h6      solo=-0.0042  LOCO_marginal=-0.0138
  sbc_dilution_acceleration_supply_drift               solo=  n/a  LOCO_marginal=+0.0000
  cashflow_quality_confirmed_surprise_drift            solo=  n/a  LOCO_marginal=+0.0000
  fresh_cashflow_confirmed_surprise_drift              solo=  n/a  LOCO_marginal=+0.0000
  cashbacked_asymmetric_surprise_drift_h6              solo=  n/a  LOCO_marginal=+0.0000
  cash_realization_surprise_disagreement               solo=  n/a  LOCO_marginal=+0.0000
  cashcycle_ambiguity_operating_fade                   solo=  n/a  LOCO_marginal=+0.0000
  episode_coherence_fresh_revision_gap                 solo=  n/a  LOCO_marginal=+0.0000
  fundamental_peer_surprise_transmission_gap           solo=  n/a  LOCO_marginal=+0.0000
  sector_shock_distributed_lag_unpaid_response         solo=  n/a  LOCO_marginal=+0.0000
  stabilized_convex_common_shock_overshoot             solo=  n/a  LOCO_marginal=+0.0000
  directed_laggraph_liquidity_unpaid_response          solo=  n/a  LOCO_marginal=+0.0000
  shrunk_directed_lag_liquidity_gap                    solo=  n/a  LOCO_marginal=+0.0000
  sector_hawkes_compensator_late_follower              solo=  n/a  LOCO_marginal=+0.0000
  signed_graph_curvature_residual_diffusion            solo=  n/a  LOCO_marginal=+0.0000
  factor_premium_dispersion_regime_transfer            solo=  n/a  LOCO_marginal=+0.0000
  intraday_supply_curve_overshoot_reversal             solo=  n/a  LOCO_marginal=+0.0000
  report_assimilation_release_continuation_h6          solo=  n/a  LOCO_marginal=+0.0000
  sector_shrunk_mode_follower_relay                    solo=  n/a  LOCO_marginal=+0.0000
  fresh_cashfunded_upgrade_wedge                       solo=  n/a  LOCO_marginal=+0.0000
  tail_survival_capacity_impedance_drift               solo=  n/a  LOCO_marginal=+0.0000
  event_freshness_accounting_confirmed_surprise        solo=  n/a  LOCO_marginal=+0.0000
  partial_graph_peer_innovation_catchup                solo=  n/a  LOCO_marginal=+0.0000
  quality_regime_conditioned_issuer_distribution       solo=  n/a  LOCO_marginal=+0.0000
  fresh_expectation_delivery_decay                     solo=  n/a  LOCO_marginal=+0.0000
  growth_expectation_convexity_break                   solo=  n/a  LOCO_marginal=+0.0000
  growth_expectation_fresh_delivery_break              solo=  n/a  LOCO_marginal=+0.0000
  fundamental_momentum_extrapolation_fragility         solo=  n/a  LOCO_marginal=+0.0000
  cashflow_deterioration_idio_downside_overshoot       solo=  n/a  LOCO_marginal=+0.0000
  tail_survival_forcedflow_reversal_h6                 solo=  n/a  LOCO_marginal=+0.0000
  downside_excitation_fragility_acceleration_h6        solo=  n/a  LOCO_marginal=+0.0000
  downside_persistence_fragility_exhaustion            solo=  n/a  LOCO_marginal=+0.0000
  resilient_downside_excitation_cooling_continuati     solo=  n/a  LOCO_marginal=+0.0000
  asymmetric_overnight_range_wedge_rebound             solo=  n/a  LOCO_marginal=+0.0000
  revision_financing_gated_idio_downside_polarity      solo=  n/a  LOCO_marginal=+0.0000
  scheduled_downside_cooling_information_drift         solo=  n/a  LOCO_marginal=+0.0000
  downside_tail_dependence_unpaid_bridge               solo=  n/a  LOCO_marginal=+0.0000
  robust_tail_equity_supply_reversal                   solo=  n/a  LOCO_marginal=+0.0000
  cashcycle_replenishment_rebound_completion           solo=  n/a  LOCO_marginal=+0.0000
  idiosyncratic_tail_resilience_absorption_repair      solo=  n/a  LOCO_marginal=+0.0000
  report_clock_tail_network_derisking_gate             solo=  n/a  LOCO_marginal=+0.0000
  anticipated_report_peer_tail_impact_gate             solo=  n/a  LOCO_marginal=+0.0000
  volume_clock_signed_flow_schedule_h6                 solo=  n/a  LOCO_marginal=+0.0000
  surprise_novelty_absorption_drift                    solo=  n/a  LOCO_marginal=+0.0000
  industry_turnover_gated_idio_repair                  solo=  n/a  LOCO_marginal=+0.0000
  latent_impact_absorption_persistence                 solo=  n/a  LOCO_marginal=+0.0000
  fractional_propagator_flow_debt_reversal             solo=  n/a  LOCO_marginal=+0.0000
  sector_relative_opening_auction_rejection_h6         solo=  n/a  LOCO_marginal=+0.0000
  decelerating_execution_debt_reversal                 solo=  n/a  LOCO_marginal=+0.0000

=== BOOK DIVERSITY (dev window) ===
mean |pairwise corr| = 0.193 | median = 0.155 | max = 0.726
most correlated pair: return_volume_recurrence_intensity_accel ~ directional_recurrence_saturation_comple (rho=+0.726)
participation ratio (effective independent factors): 7.5 / 14

=== ARCHIVE DIAGNOSTICS (recorded by the run) ===
  factor_id                                            IS_IC   degrad  max|corr| coverage
  sbc_dilution_acceleration_supply_drift               -0.002    None  0.302   1.00
  cashflow_quality_confirmed_surprise_drift            -0.003    None  0.268   1.00
  fresh_cashflow_confirmed_surprise_drift              +0.019  +0.45  0.262   1.00
  cashbacked_asymmetric_surprise_drift_h6              +0.014  -0.51  0.262   1.00
  cash_realization_surprise_disagreement               +0.000    None  0.268   1.00
  cashcycle_ambiguity_operating_fade                   -0.007  +0.12  0.331   1.00
  episode_coherence_fresh_revision_gap                 -0.006  +1.29  0.144   1.00
  fundamental_peer_surprise_transmission_gap           +0.007  +2.26  0.581   1.00
  scheduled_flow_impact_decay_reversal                 -0.012  +2.74  0.591   1.00
  sector_shock_distributed_lag_unpaid_response         +0.007  +1.08  0.218   1.00
  stabilized_convex_common_shock_overshoot             -0.009  +0.60  0.502   1.00
  directed_laggraph_liquidity_unpaid_response          +0.004    None  0.415   1.00
  shrunk_directed_lag_liquidity_gap                    +0.010  +0.17  0.422   1.00
  return_volume_recurrence_intensity_acceleration      -0.017  +2.89  0.770   1.00
  directional_recurrence_saturation_completion         -0.018  +2.09  0.770   1.00
  sector_hawkes_compensator_late_follower              +0.017  +1.97  0.446   1.00
  signed_graph_curvature_residual_diffusion            +0.009  +1.33  0.492   1.00
  factor_premium_dispersion_regime_transfer            -0.003    None  0.272   1.00
  relaxation_time_orderflow_persistence_drift          -0.020  +0.96  0.632   0.93
  persistent_liquidity_commonality_derisking           -0.006  +2.27  0.337   1.00
  intraday_supply_curve_overshoot_reversal             +0.016  +0.60  0.318   1.00
  thin_side_sqrt_impact_impulse_h6                     -0.011  +1.41  0.575   1.00
  completed_marked_arrival_absorption_reversal_h6      -0.005  +1.46  0.205   1.00
  report_assimilation_release_continuation_h6          -0.001    None  0.203   1.00
  idio_thin_side_vacuum_reversal_h6                    +0.004    None  0.473   1.00
  sector_shrunk_mode_follower_relay                    +0.010  +1.65  0.317   1.00
  fresh_cashfunded_upgrade_wedge                       +0.005  +1.77  0.236   1.00
  tail_survival_capacity_impedance_drift               -0.007  -3.39  0.662   1.00
  event_freshness_accounting_confirmed_surprise        -0.006  -0.25  0.331   1.00
  partial_graph_peer_innovation_catchup                +0.017  +1.16  0.623   1.00
  quality_regime_conditioned_issuer_distribution       +0.005  +8.20  0.167   1.00
  fresh_expectation_delivery_decay                     +0.007  +0.59  0.278   1.00
  growth_expectation_convexity_break                   +0.000    None  0.019   1.00
  growth_expectation_fresh_delivery_break              +0.005  +0.67  0.115   1.00
  fundamental_momentum_extrapolation_fragility         +0.003    None  0.014   1.00
  downside_pressure_freshness_h6                       +0.015  +1.01  0.632   1.00
  cashflow_deterioration_idio_downside_overshoot       +0.010  +0.91  0.575   1.00
  tail_survival_forcedflow_reversal_h6                 -0.002    None  0.662   1.00
  superlinear_downside_variance_relaxation_tilt        +0.003    None  0.074   1.00
  downside_excitation_fragility_acceleration_h6        +0.010  +1.46  0.346   1.00
  downside_persistence_fragility_exhaustion            +0.008  +0.78  0.311   1.00
  resilient_downside_excitation_cooling_continuati     +0.016  +0.50  0.506   1.00
  prompt_cooling_thin_traversal_reversal               +0.017  +0.95  0.461   1.00
  asymmetric_overnight_range_wedge_rebound             +0.009  +2.92  0.609   0.99
  revision_financing_gated_idio_downside_polarity      +0.003    None  0.274   1.00
  downside_reflexive_probe_persistence                 +0.013  +3.06  0.591   0.96
  scheduled_downside_cooling_information_drift         +0.005    None  0.061   1.00
  downside_tail_dependence_unpaid_bridge               -0.018  +1.08  0.246   1.00
  robust_tail_equity_supply_reversal                   +0.013  +1.43  0.586   1.00
  cashcycle_replenishment_rebound_completion           +0.001    None  0.238   1.00
  idiosyncratic_tail_resilience_absorption_repair      +0.013  +1.07  0.326   1.00
  report_clock_tail_network_derisking_gate             -0.007  +1.29  0.196   1.00
  anticipated_report_peer_tail_impact_gate             +0.005  +1.89  0.236   1.00
  volume_clock_signed_flow_schedule_h6                 -0.023  +0.98  0.525   1.00
  calendar_phase_metaorder_impact_deficit              -0.015  +2.67  0.243   1.00
  persistent_participation_pressure_continuation       -0.012  +3.76  0.493   1.00
  surprise_novelty_absorption_drift                    +0.012  +0.44  0.235   1.00
  industry_turnover_gated_idio_repair                  +0.008  +1.35  0.609   1.00
  latent_impact_absorption_persistence                 -0.005    None  0.241   1.00
  fractional_propagator_flow_debt_reversal             +0.018  -0.48  0.623   1.00
  sector_relative_opening_auction_rejection_h6         +0.000    None  0.000   1.00
  decelerating_execution_debt_reversal                 +0.011  +0.06  0.379   1.00
```
