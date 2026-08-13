# L2WFP_terra_s0 — WF factor analysis

Factors: 27 computed / 27 in book (0 failed). Fit window < 2021-07-20; OOS = 10 prequential 126-bar blocks (2021-07 -> 2026-07). All ICs are pooled per-underlying; the WF statistic is the MEAN of the per-block ICs.

- per-factor mean |IC| fit(blocks): 0.0137 -> WF: 0.0107
- sign-consistent retention (median): 0.14
- factors with WF hit-rate >= 0.6: 11/27
- best WF factor: bayesian_candle_absorption_pressure_j (blockmean 0.0569)

| model | n | IS blockmean | WF blockmean | WF std | hit |
|---|---|---|---|---|---|
| ridge | 27 | 0.0733 | 0.0494 | 0.0125 | 100% |
| lasso | 27 | 0.0745 | 0.0542 | 0.0151 | 100% |
| lightgbm | 27 | 0.1446 | 0.0497 | 0.0276 | 90% |
