# Evolution timing diagnostic

- Wall clock: **5.6m**
- Candidates scored: **21** (generations=3, n_trials=21, archive=10)
- Model fits: **608** (~29.0/candidate)
- LLM cost: **$0.0277** of $3.00 budget (26 calls)

## Phases

| phase | time |
|---|---|
| panel load (once) | 2.87s |
| seeding | 56.61s |
| evaluation (total) | 2.4m |

## LLM by role

| role | calls | time | in_tok | out_tok | cost $ |
|---|--:|--:|--:|--:|--:|
| mutate | 12 | 1.4m | 72449 | 5811 | 0.0144 |
| crossover | 5 | 47.88s | 33843 | 2735 | 0.0067 |
| codegen | 8 | 42.65s | 25859 | 2807 | 0.0056 |
| brainstorm | 1 | 12.55s | 3306 | 899 | 0.0010 |

## Evaluation metrics (self-time, additive)

| metric | self | inclusive | calls | ms/call |
|---|--:|--:|--:|--:|
| _combined_prediction | 2.4m | 2.4m | 608 | 232.0 |
| _pooled_ic | 669ms | 669ms | 1111 | 0.6 |
| _independence | 208ms | 208ms | 21 | 9.9 |
| _residual_ic | 156ms | 158ms | 21 | 7.5 |
| _structural_novelty | 70ms | 70ms | 21 | 3.3 |
| _behavior_descriptors | 62ms | 81ms | 21 | 3.9 |
| _turnover_netcost | 59ms | 59ms | 21 | 2.8 |
| evaluate_candidate | 20ms | 2.4m | 21 | 6778.7 |
| _stress_mask | 19ms | 19ms | 21 | 0.9 |
| _refit_cpcv_scores | 18ms | 2.1m | 21 | 5998.3 |
| _robustness | 6ms | 2.1m | 21 | 6006.2 |
| _coverage | 2ms | 2ms | 21 | 0.1 |
| _standalone_cpcv_ics | 1ms | 141ms | 21 | 6.7 |
| _marginal_value | 1ms | 15.54s | 21 | 739.9 |
| _zoo_dedup | 0ms | 0ms | 21 | 0.0 |

## Eval cost vs book size

| book size | n | mean eval |
|---|--:|--:|
| 00-04 | 11 | 6.20s |
| 05-09 | 9 | 7.55s |
| 10-19 | 1 | 7.74s |
