# L4IC_terra_s0 — WF factor analysis

Factors: 8 computed / 8 in book (0 failed). Fit window < 2021-07-20; OOS = 10 prequential 126-bar blocks (2021-07 -> 2026-07). All ICs are pooled per-underlying; the WF statistic is the MEAN of the per-block ICs.

- per-factor mean |IC| fit(blocks): 0.0429 -> WF: 0.0292
- sign-consistent retention (median): 0.59
- factors with WF hit-rate >= 0.6: 2/8
- best WF factor: concentrated_flow_price_inertia (blockmean 0.0657)

| model | n | IS blockmean | WF blockmean | WF std | hit |
|---|---|---|---|---|---|
| ridge | 8 | 0.1055 | 0.0817 | 0.0394 | 100% |
| lasso | 8 | 0.1053 | 0.0817 | 0.0394 | 100% |
| lightgbm | 8 | 0.1498 | 0.0521 | 0.0255 | 100% |
