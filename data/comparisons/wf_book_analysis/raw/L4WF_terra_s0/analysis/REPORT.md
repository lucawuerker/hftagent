# L4WF_terra_s0 — WF factor analysis

Factors: 57 computed / 57 in book (0 failed). Fit window < 2021-07-20; OOS = 10 prequential 126-bar blocks (2021-07 -> 2026-07). All ICs are pooled per-underlying; the WF statistic is the MEAN of the per-block ICs.

- per-factor mean |IC| fit(blocks): 0.0186 -> WF: 0.0179
- sign-consistent retention (median): 0.80
- factors with WF hit-rate >= 0.6: 35/57
- best WF factor: sell_arrival_impact_compression_repair (blockmean 0.0709)

| model | n | IS blockmean | WF blockmean | WF std | hit |
|---|---|---|---|---|---|
| ridge | 57 | 0.0888 | 0.0642 | 0.0233 | 100% |
| lasso | 57 | 0.0822 | 0.0569 | 0.0283 | 100% |
| lightgbm | 57 | 0.1999 | 0.0650 | 0.0491 | 80% |
