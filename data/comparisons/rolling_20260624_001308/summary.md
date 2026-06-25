# Rolling-window factor-set comparison

- runs aggregated: **279** (ticker × OOS-month)

- tickers: CORN, CPER, FXE, FXI, GLD, IEF, QQQ, SHY, SPY, USO


## Mean OOS Sharpe across all runs & models, by factor set

- `gpt5.4mini120650` = 9.1858
- `gpt4omini120650` = 7.0653
- `main` = 6.1228


## Mean OOS Sharpe by factor set × model

| prerun | elastic_net | ensemble | gradient_boosting | lasso | lightgbm | linear_regression | random_forest | ridge | xgboost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt4omini120650 | 16.776 | 8.259 | 1.336 | 16.762 | 2.424 | 7.567 | 8.761 | 7.941 | 2.525 |
| gpt5.4mini120650 | 13.310 | 11.338 | 2.899 | 13.767 | 5.356 | 8.955 | 13.719 | 8.941 | 7.605 |
| main | 11.375 | 6.917 | 3.594 | 11.610 | 3.701 | 5.915 | 6.196 | 6.227 | 4.115 |


## Outputs

- `combined/` — `bruteforce_all.csv`, `importance_all.csv`, `diversity_all.csv`, `ic_all.csv` (every run, tagged with `ticker, oos_month, is_window`).

- `per_ticker/<ticker>/` — `importance_over_months__<prerun>__<model>.csv` (+ heatmap PNG) and `performance_<metric>__<model>.png`.

- `runs/<ticker>/<oos_month>/` — each window's full comparison (`report.md`, figures, `run.log`).
