# Evolution timing diagnostic

- Wall clock: **3.6m**
- Candidates scored: **8** (generations=2, n_trials=8, archive=0)
- Model fits: **8** (~1.0/candidate)
- LLM cost: **$0.0257** of $1.00 budget (20 calls)

## Phases

| phase | time |
|---|---|
| panel load (once) | 1.0m |
| seeding | 1.2m |
| evaluation (total) | 1.3m |

## LLM by role

| role | calls | time | in_tok | out_tok | cost $ |
|---|--:|--:|--:|--:|--:|
| codegen | 11 | 59.98s | 68196 | 4234 | 0.0128 |
| mutate | 5 | 43.08s | 35129 | 3111 | 0.0071 |
| crossover | 3 | 20.93s | 23287 | 1914 | 0.0046 |
| brainstorm | 1 | 11.92s | 3604 | 1088 | 0.0012 |

## Evaluation metrics (self-time, additive)

| metric | self | inclusive | calls | ms/call |
|---|--:|--:|--:|--:|
| _combined_prediction | 11.49s | 11.49s | 8 | 1436.5 |
| _pooled_ic | 359ms | 359ms | 86 | 4.2 |
| _turnover_netcost | 232ms | 232ms | 8 | 29.0 |
| _independence | 45ms | 45ms | 8 | 5.6 |
| _structural_novelty | 12ms | 12ms | 8 | 1.5 |
| _coverage | 8ms | 8ms | 8 | 1.0 |
| evaluate_candidate | 6ms | 12.16s | 8 | 1519.4 |
| _marginal_penalties | 2ms | 52ms | 8 | 6.5 |
| _marginal_value | 0ms | 11.53s | 8 | 1441.0 |
| _residual_ic | 0ms | 58ms | 8 | 7.3 |
| _apply_marginal_penalties | 0ms | 0ms | 8 | 0.0 |
| _zoo_dedup | 0ms | 0ms | 8 | 0.0 |

## Eval cost vs book size

| book size | n | mean eval |
|---|--:|--:|
| 00-04 | 8 | 9.57s |
