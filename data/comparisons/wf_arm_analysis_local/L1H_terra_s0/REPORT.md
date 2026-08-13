# L1H_terra_s0 — WF factor analysis

Factors: 22 computed / 22 in book (0 failed). Fit window < 2021-07-20; OOS = 10 prequential 126-bar blocks (2021-07 -> 2026-07). All ICs are pooled per-underlying; the WF statistic is the MEAN of the per-block ICs.

- per-factor mean |IC| fit(blocks): 0.0179 -> WF: 0.0153
- sign-consistent retention (median): 0.58
- factors with WF hit-rate >= 0.6: 11/21
- best WF factor: corporate_financing_footprint_persistence (blockmean 0.0636)

| model | n | IS blockmean | WF blockmean | WF std | hit |
|---|---|---|---|---|---|
| ridge | 22 | 0.0620 | 0.0415 | 0.0197 | 100% |
| lasso | 22 | -0.0000 | 0.0000 | 0.0000 | 40% |
| lightgbm | 22 | 0.1409 | 0.0390 | 0.0324 | 80% |
