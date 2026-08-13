# L2WFB_terra_s0 — WF factor analysis

Factors: 13 computed / 13 in book (0 failed). Fit window < 2021-07-20; OOS = 10 prequential 126-bar blocks (2021-07 -> 2026-07). All ICs are pooled per-underlying; the WF statistic is the MEAN of the per-block ICs.

- per-factor mean |IC| fit(blocks): 0.0195 -> WF: 0.0216
- sign-consistent retention (median): 0.87
- factors with WF hit-rate >= 0.6: 8/13
- best WF factor: causal_return_volume_analog_regime_forecast (blockmean 0.1450)

| model | n | IS blockmean | WF blockmean | WF std | hit |
|---|---|---|---|---|---|
| ridge | 13 | 0.0990 | 0.0882 | 0.0166 | 100% |
| lasso | 13 | 0.1051 | 0.1020 | 0.0158 | 100% |
| lightgbm | 13 | 0.1479 | 0.0387 | 0.0483 | 70% |
