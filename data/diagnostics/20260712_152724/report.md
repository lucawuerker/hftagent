# Evolution timing diagnostic

- Wall clock: **1.2m**
- Candidates scored: **5** (generations=1, n_trials=5, archive=3)
- Model fits: **124** (~24.8/candidate)
- LLM cost: **$0.0068** of $3.00 budget (7 calls)

## Phases

| phase | time |
|---|---|
| panel load (once) | 945ms |
| seeding | 34.92s |
| evaluation (total) | 11.27s |

## LLM by role

| role | calls | time | in_tok | out_tok | cost $ |
|---|--:|--:|--:|--:|--:|
| mutate | 2 | 13.47s | 11719 | 994 | 0.0024 |
| codegen | 3 | 24.21s | 9468 | 1261 | 0.0022 |
| crossover | 1 | 9.81s | 6772 | 654 | 0.0014 |
| brainstorm | 1 | 8.02s | 3306 | 542 | 0.0008 |

## Evaluation metrics (self-time, additive)

| metric | self | inclusive | calls | ms/call |
|---|--:|--:|--:|--:|
| _combined_prediction | 9.97s | 9.97s | 124 | 80.4 |
| _pooled_ic | 108ms | 108ms | 249 | 0.4 |
| _independence | 77ms | 77ms | 5 | 15.4 |
| _behavior_descriptors | 20ms | 27ms | 5 | 5.4 |
| _turnover_netcost | 16ms | 16ms | 5 | 3.2 |
| _stress_mask | 7ms | 7ms | 5 | 1.4 |
| _structural_novelty | 6ms | 6ms | 5 | 1.2 |
| _residual_ic | 6ms | 7ms | 5 | 1.4 |
| evaluate_candidate | 6ms | 10.22s | 5 | 2044.5 |
| _refit_cpcv_scores | 3ms | 7.03s | 5 | 1405.9 |
| _robustness | 2ms | 7.06s | 5 | 1412.0 |
| _coverage | 1ms | 1ms | 5 | 0.1 |
| _standalone_cpcv_ics | 0ms | 25ms | 5 | 5.0 |
| _marginal_value | 0ms | 3.01s | 5 | 601.8 |
| _zoo_dedup | 0ms | 0ms | 5 | 0.0 |

## Eval cost vs book size

| book size | n | mean eval |
|---|--:|--:|
| 00-04 | 5 | 2.25s |
