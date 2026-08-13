# L5WF_terra_s0 — WF factor analysis

Factors: 35 computed / 35 in book (0 failed). Fit window < 2021-07-20; OOS = 10 prequential 126-bar blocks (2021-07 -> 2026-07). All ICs are pooled per-underlying; the WF statistic is the MEAN of the per-block ICs.

- per-factor mean |IC| fit(blocks): 0.0187 -> WF: 0.0182
- sign-consistent retention (median): 0.53
- factors with WF hit-rate >= 0.6: 13/30
- best WF factor: correlation_release_resilient_repair_lag (blockmean 0.0685)

| model | n | IS blockmean | WF blockmean | WF std | hit |
|---|---|---|---|---|---|
| ridge | 35 | 0.0612 | 0.0384 | 0.0171 | 100% |
| lasso | 35 | 0.0588 | 0.0379 | 0.0175 | 90% |
| lightgbm | 35 | 0.1571 | 0.0485 | 0.0379 | 90% |
