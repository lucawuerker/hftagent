# L7WF_terra_s0 — WF factor analysis

Factors: 42 computed / 42 in book (0 failed). Fit window < 2021-07-20; OOS = 10 prequential 126-bar blocks (2021-07 -> 2026-07). All ICs are pooled per-underlying; the WF statistic is the MEAN of the per-block ICs.

- per-factor mean |IC| fit(blocks): 0.0216 -> WF: 0.0240
- sign-consistent retention (median): 0.85
- factors with WF hit-rate >= 0.6: 16/36
- best WF factor: surprise_impact_saturation_phase_switch (blockmean 0.1112)

| model | n | IS blockmean | WF blockmean | WF std | hit |
|---|---|---|---|---|---|
| ridge | 42 | 0.0879 | 0.0559 | 0.0215 | 100% |
| lasso | 42 | 0.0799 | 0.0496 | 0.0278 | 100% |
| lightgbm | 42 | 0.1691 | 0.0391 | 0.0478 | 70% |
