# L2WF Terra s0 (walk-forward ladder)

Generated 2026-08-03 from run artifacts (lineage, gen_quality, prequential, state, factor DB, usage).
TEST tail untouched — all numbers are dev-window or honest prequential OOS.

```
=== 1. RUN OVERVIEW ===
trials: 758 | kept_pool: 756 | final archive: 19
usage by role:
  brainstorm   calls=   2 in= 0.01M out= 0.00M $   0.11 errors=0
  codegen      calls=  13 in= 0.08M out= 0.03M $   0.67 errors=0
  crossover    calls= 336 in= 5.15M out= 0.92M $  26.61 errors=0
  mutation     calls= 857 in=11.99M out= 2.41M $  66.05 errors=0
  TOTAL        calls=1208 $93.43 errors=0

=== 2. OPERATOR MIX (billed candidates) ===
  llm_semantic           n= 430 selectable= 430 mean_marginal=-0.0029 max=+0.0441
  crossover              n= 222 selectable= 222 mean_marginal=-0.0042 max=+0.0359
  jitter                 n=  53 selectable=  53 mean_marginal=-0.0011 max=+0.0852
  llm_semantic_creative  n=  41 selectable=  41 mean_marginal=-0.0019 max=+0.0173
  seed                   n=  12 selectable=  11 mean_marginal=-0.0009 max=+0.0175

=== 3. GENERATION TRAJECTORY ===
  gen  billed  archive kept_pool mean_mv   max_mv   novelty evict
    0     12       8       11 +0.0029  +0.0175  0.853    2
    1     45      21       56 +0.0013  +0.0109  0.818    7
    2     40      26       96 +0.0012  +0.0110  0.826   14
    3     44      27      140 +0.0032  +0.0141  0.837   12
    4     38      18      178 +0.0054  +0.0142  0.846   14
    5     38      22      216 +0.0033  +0.0131  0.854    6
    6     38      19      254 +0.0009  +0.0113  0.866   11
    7     42      24      296 -0.0001  +0.0125  0.856    5
    8     36      20      332 -0.0031  +0.0092  0.843   13
    9     37      18      369 -0.0015  +0.0118  0.880    9
   10     30      15      399 +0.0052  +0.0441  0.899    7
   11     36      18      435 +0.0042  +0.0554  0.848    4
   12     37      19      472 +0.0045  +0.0625  0.832    6
   13     36      19      508 -0.0000  +0.0124  0.863   16
   14     31      21      539 -0.0011  +0.0359  0.856   11
   15     42      16      581 +0.0023  +0.0132  0.868    8
   16     41      19      622 +0.0059  +0.0852  0.866    3
   17     30      17      652 +0.0013  +0.0193  0.839    5
   18     33      16      685 +0.0020  +0.0331  0.850    7
   19     35      22      720 +0.0007  +0.0426  0.847    2
   20     37      19      756 -0.0007  +0.0120  0.839    7

=== 4. PREQUENTIAL (honest OOS on never-seen blocks) ===
  idx gen  window                        combined_OOS_IC   PBO    n_obs archive
    0    1  2015-03-16 -> 2015-10-29   +0.0447      0.5    15485   8
    1    2  2015-10-29 -> 2016-06-17   +0.0195      0.18571428571428572    15886   21
    2    3  2016-06-17 -> 2017-02-03   +0.0327      0.34285714285714286    16018   26
    3    4  2017-02-03 -> 2017-09-21   +0.0011      0.2714285714285714    16111   27
    4    5  2017-09-21 -> 2018-05-10   +0.0269      0.14285714285714285    16072   18
    5    6  2018-05-10 -> 2018-12-27   -0.0345      0.3142857142857143    16034   22
    6    7  2018-12-27 -> 2019-08-15   +0.0618      0.15714285714285714    16337   19
    7    8  2019-08-15 -> 2020-04-02   -0.0031      0.44285714285714284    16437   24
    8    9  2020-04-02 -> 2020-11-17   +0.0281      0.35714285714285715    16500   20
    9   10  2020-11-17 -> 2021-07-20   +0.0209      0.2571428571428571    17177   18
    10   11  2021-07-20 -> 2022-01-18   +0.0249      0.24285714285714285    12908   15
    11   12  2022-01-18 -> 2022-07-20   +0.1178      0.14285714285714285    12952   18
    12   13  2022-07-20 -> 2023-01-19   -0.0794      0.17142857142857143    12910   19
    13   14  2023-01-19 -> 2023-07-21   +0.0306      0.3142857142857143    12834   19
    14   15  2023-07-21 -> 2024-01-22   -0.0146      0.3    12810   21
    15   16  2024-01-22 -> 2024-07-23   +0.0455      0.2857142857142857    12833   16
    16   17  2024-07-23 -> 2025-01-23   -0.0001      0.4    12828   19
    17   18  2025-01-23 -> 2025-07-25   +0.0609      0.14285714285714285    12835   17
    18   19  2025-07-25 -> 2026-01-26   -0.0321      0.21428571428571427    12809   16
    19   20  2026-01-26 -> 2026-07-27   +0.0428      0.14285714285714285    12192   22
  mean=+0.0197 median=+0.0259 min=-0.0794 max=+0.1178 positive=14/20

=== 5. FINAL ARCHIVE (the 19-factor book) ===
  factor_id                                          grp gen op            marginal indep   pars novelty
  orthogonal_volume_relaxation_displacement_revers   0  15 llm_semantic +0.0120 +0.0288  -102 0.848
  cash_confirmed_margin_step_diffusion_v2            0  13 llm_semantic +0.0069 +0.0136   -52 0.866
  cash_return_accrual_roic_residual                  0   1 llm_semantic +0.0060 -0.0063   -19 0.640
  surprise_credit_extension_trap_j_2                 0  20 jitter       +0.0059 -0.0110   -49 0.766
  renewal_inventory_delayed_transfer_state           0  20 llm_semantic +0.0055 +0.0038  -179 0.890
  rolling_transfer_deficit_signal                    0  18 llm_semantic +0.0040 +0.0111   -76 0.898
  muted_productivity_innovation_assimilation         0  19 llm_semantic +0.0024 -0.0008   -44 0.877
  estimate_revision_relative_reaction_gap            0  16 llm_semantic +0.0013 +0.0010   -31 0.827
  trade_credit_optionality_unpriced_response         0  19 llm_semantic +0.0007 -0.0046   -46 0.884
  deleveraging_validated_earnings_attention_gap      0  14 crossover    -0.0000 +0.0049   -87 0.913
  silent_acceptance_credit_cash_audit_unwind         0  14 llm_semantic -0.0010 +0.0138  -109 0.898
  cash_collected_sales_acceleration_lag              0   5 llm_semantic -0.0010 +0.0064   -40 0.885
  report_congestion_cash_conversion_resolution       0  16 llm_semantic -0.0017 +0.0021   -65 0.886
  liquidation_sourced_cash_roic_correction           0  19 llm_semantic -0.0021 +0.0086   -34 0.640
  incremental_roic_filing_diffusion                  0  18 llm_semantic -0.0027 +0.0004   -19 0.750
  validated_trade_credit_capacity_diffusion          0  19 llm_semantic -0.0038 +0.0143   -57 0.884
  working_capital_harvest_disclosure_decay           0  20 llm_semantic -0.0067 +0.0146   -45 0.829
  cash_validated_efficiency_transfer_diffusion_j     0  20 jitter       -0.0129 +0.0090  -124 0.910
  quiet_operating_leverage_delivery_drift            0  12 llm_semantic -0.0264 +0.0153   -78 0.852
  book age distribution (generation born): {1: 1, 5: 1, 12: 1, 13: 1, 14: 2, 15: 1, 16: 2, 18: 2, 19: 4, 20: 4}
  book mechanism-group distribution: {0: 19}
  categories: {'other': 10, 'sentiment': 4, 'microstructure': 3, 'statistical_arbitrage': 1, 'momentum': 1}
```

```
dev window: 2010-01-04 -> 2026-07-27 (4165 bars); TEST tail untouched
IS bars=3937 VAL bars=228

=== COMBINED MODEL (fit IS, scored VAL) ===
LightGBM combined VAL IC (19 factors): +0.0075
Ridge    combined VAL IC (19 factors): +0.0034

=== PER-FACTOR: solo VAL IC | LOCO marginal on final book ===
  cash_confirmed_margin_step_diffusion_v2              solo=-0.0269  LOCO_marginal=+0.0118
  validated_trade_credit_capacity_diffusion            solo=-0.0222  LOCO_marginal=+0.0061
  orthogonal_volume_relaxation_displacement_revers     solo=+0.0270  LOCO_marginal=+0.0061
  renewal_inventory_delayed_transfer_state             solo=+0.0025  LOCO_marginal=+0.0055
  cash_collected_sales_acceleration_lag                solo=-0.0125  LOCO_marginal=+0.0050
  report_congestion_cash_conversion_resolution         solo=-0.0041  LOCO_marginal=+0.0048
  silent_acceptance_credit_cash_audit_unwind           solo=+0.0076  LOCO_marginal=+0.0044
  deleveraging_validated_earnings_attention_gap        solo=+0.0082  LOCO_marginal=+0.0027
  muted_productivity_innovation_assimilation           solo=-0.0097  LOCO_marginal=+0.0024
  surprise_credit_extension_trap_j_2                   solo=-0.0008  LOCO_marginal=+0.0010
  incremental_roic_filing_diffusion                    solo=-0.0285  LOCO_marginal=+0.0006
  working_capital_harvest_disclosure_decay             solo=+0.0031  LOCO_marginal=+0.0004
  estimate_revision_relative_reaction_gap              solo=+0.0173  LOCO_marginal=-0.0001
  rolling_transfer_deficit_signal                      solo=+0.0301  LOCO_marginal=-0.0004
  liquidation_sourced_cash_roic_correction             solo=+0.0149  LOCO_marginal=-0.0008
  trade_credit_optionality_unpriced_response           solo=-0.0182  LOCO_marginal=-0.0015
  cash_return_accrual_roic_residual                    solo=+0.0149  LOCO_marginal=-0.0037
  cash_validated_efficiency_transfer_diffusion_j       solo=+0.0073  LOCO_marginal=-0.0047
  quiet_operating_leverage_delivery_drift              solo=-0.0459  LOCO_marginal=-0.0210

=== BOOK DIVERSITY (dev window) ===
mean |pairwise corr| = 0.062 | median = 0.018 | max = 0.909
most correlated pair: cash_return_accrual_roic_residual ~ liquidation_sourced_cash_roic_correction (rho=+0.909)
participation ratio (effective independent factors): 14.1 / 19

=== ARCHIVE DIAGNOSTICS (recorded by the run) ===
  factor_id                                            IS_IC   degrad  max|corr| coverage
  cash_return_accrual_roic_residual                    +0.007  +1.73  0.909   1.00
  cash_collected_sales_acceleration_lag                +0.005  -2.72  0.189   1.00
  quiet_operating_leverage_delivery_drift              +0.008  -4.30  0.048   1.00
  cash_confirmed_margin_step_diffusion_v2              +0.011  -1.62  0.130   1.00
  deleveraging_validated_earnings_attention_gap        +0.004    None  0.163   1.00
  silent_acceptance_credit_cash_audit_unwind           +0.005    None  0.054   1.00
  orthogonal_volume_relaxation_displacement_revers     +0.032  +0.85  0.329   1.00
  estimate_revision_relative_reaction_gap              +0.018  +0.79  0.548   1.00
  report_congestion_cash_conversion_resolution         +0.003    None  0.259   1.00
  incremental_roic_filing_diffusion                    +0.005    None  0.581   1.00
  rolling_transfer_deficit_signal                      +0.015  +2.10  0.329   1.00
  muted_productivity_innovation_assimilation           +0.005    None  0.581   1.00
  validated_trade_credit_capacity_diffusion            +0.002    None  0.159   1.00
  liquidation_sourced_cash_roic_correction             +0.008  +1.67  0.909   1.00
  trade_credit_optionality_unpriced_response           -0.001    None  0.415   1.00
  renewal_inventory_delayed_transfer_state             +0.008  +0.13  0.047   1.00
  working_capital_harvest_disclosure_decay             +0.010  +0.18  0.530   1.00
  surprise_credit_extension_trap_j_2                   -0.005  +1.74  0.300   1.00
  cash_validated_efficiency_transfer_diffusion_j       +0.015  -0.97  0.381   1.00
```
