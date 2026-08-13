# L2WF_terra_s0 — WF factor analysis

Factors: 19 computed / 19 in book (0 failed). Fit window < 2021-07-20; OOS = 10 prequential 126-bar blocks (2021-07 -> 2026-07). All ICs are pooled per-underlying; the WF statistic is the MEAN of the per-block ICs.

- per-factor mean |IC| fit(blocks): 0.0151 -> WF: 0.0158
- sign-consistent retention (median): 0.85
- factors with WF hit-rate >= 0.6: 10/19
- best WF factor: rolling_transfer_deficit_signal (blockmean 0.0580)

| model | n | IS blockmean | WF blockmean | WF std | hit |
|---|---|---|---|---|---|
| ridge | 19 | 0.0660 | 0.0532 | 0.0259 | 100% |
| lasso | 19 | 0.0641 | 0.0551 | 0.0265 | 100% |
| lightgbm | 19 | 0.1763 | 0.0152 | 0.0797 | 60% |
