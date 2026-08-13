# L4D_terra_s0 — WF factor analysis

Factors: 46 computed / 46 in book (0 failed). Fit window < 2021-07-20; OOS = 10 prequential 126-bar blocks (2021-07 -> 2026-07). All ICs are pooled per-underlying; the WF statistic is the MEAN of the per-block ICs.

- per-factor mean |IC| fit(blocks): 0.0184 -> WF: 0.0187
- sign-consistent retention (median): 0.86
- factors with WF hit-rate >= 0.6: 18/46
- best WF factor: microcap_highwater_path_validation_j_1_j_j_4 (blockmean 0.1633)

| model | n | IS blockmean | WF blockmean | WF std | hit |
|---|---|---|---|---|---|
| ridge | 46 | 0.0784 | 0.0541 | 0.0121 | 100% |
| lasso | 46 | 0.0764 | 0.0544 | 0.0108 | 100% |
| lightgbm | 46 | 0.1853 | 0.0334 | 0.0245 | 90% |
