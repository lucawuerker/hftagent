# Rolling factor comparison report figures

Generated files are written as both `.png` and `.pdf`.

- `fig01_ensemble_oos_sharpe_distribution.pdf` / `fig01_ensemble_oos_sharpe_distribution.png`: Distribution of ensemble OOS Sharpe values across rolling ETF-month windows. Boxes omit outlier dots to keep the central distribution visible; black diamonds show means.
- `fig02_model_robustness_median_oos_sharpe.pdf` / `fig02_model_robustness_median_oos_sharpe.png`: Median OOS Sharpe by factor set and model. This highlights whether a factor set performs only for one model class or remains useful across model specifications.
- `fig03_ensemble_oos_sharpe_over_time.pdf` / `fig03_ensemble_oos_sharpe_over_time.png`: Monthly median ensemble OOS Sharpe across ETFs, with the shaded area showing the interquartile range across ETFs.
- `fig04_ticker_factor_set_heatmap.pdf` / `fig04_ticker_factor_set_heatmap.png`: Median ensemble OOS Sharpe by ETF and factor set, summarising which universes drive the aggregate comparison.
- `fig05_oos_ic_vs_oos_sharpe.pdf` / `fig05_oos_ic_vs_oos_sharpe.png`: Relationship between OOS information coefficient and realised ensemble Sharpe. The Sharpe axis is percentile-clipped for readability.
- `fig06_factor_library_diversity.pdf` / `fig06_factor_library_diversity.png`: Factor-library diversity summary: effective factor ratio, redundancy, and mean absolute factor correlation across rolling windows.
- `fig07_top_factor_importance_recurrence.pdf` / `fig07_top_factor_importance_recurrence.png`: Most recurrent top-five factors in the model-based importance analysis. Recurrence is counted across ticker-month windows and importance models.
- `fig08_independence_vs_oos_sharpe.pdf` / `fig08_independence_vs_oos_sharpe.png`: Scatter of factor-library independence versus realised ensemble OOS Sharpe for each ticker-month and factor set.

The OOS Sharpe figures are research diagnostics from the frictionless 10-second comparison backtest.
