# run_one_strategy.py
from quant_fund_agent.backtesting.data_loader import load_panel
from quant_fund_agent.backtesting.strategy_backtester import backtest_strategy
from quant_fund_agent.factors import discover_factors, instantiate_factor
from quant_fund_agent.strategies import discover_strategies, instantiate_strategy

data = load_panel("ticker_data")

# discover registries
discover_factors()
discover_strategies()

# strategy
strategy = instantiate_strategy("alpha_blend_test")

# compute required factor signals
factor_signals = {}
for fid in strategy.factor_ids:
    f = instantiate_factor(fid)
    factor_signals[fid] = f.calc(data)

# run backtest
res = backtest_strategy(strategy, factor_signals, data)

print(res.metrics)
print(res.portfolio_returns.head())
print(res.cumulative_returns.tail())