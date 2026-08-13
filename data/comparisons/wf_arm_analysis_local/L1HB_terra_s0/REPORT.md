# L1HB_terra_s0 — WF factor analysis

Factors: 24 computed / 24 in book (0 failed). Fit window < 2021-07-20; OOS = 10 prequential 126-bar blocks (2021-07 -> 2026-07). All ICs are pooled per-underlying; the WF statistic is the MEAN of the per-block ICs.

- per-factor mean |IC| fit(blocks): 0.0109 -> WF: 0.0109
- sign-consistent retention (median): 0.76
- factors with WF hit-rate >= 0.6: 10/24
- best WF factor: consensus_revision_price_absorption_deficit (blockmean 0.0397)

| model | n | IS blockmean | WF blockmean | WF std | hit |
|---|---|---|---|---|---|
| ridge | 24 | 0.0529 | 0.0337 | 0.0189 | 90% |
| lasso | 24 | 0.0556 | 0.0390 | 0.0158 | 100% |
| lightgbm | 24 | 0.1320 | 0.0343 | 0.0282 | 90% |
