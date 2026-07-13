"""Illustrate why standalone IC is insufficient for factor-book construction.

The script downloads SPY's full daily OHLCV history from Yahoo Finance, builds
three deliberately simple factors, and evaluates them on the final 30% of the
sample.  It writes the downloaded data, a results table, and a thesis-ready
figure to ``data/thesis_examples/spy_ic_complementarity`` by default.

This is an explanatory construction, not a claim that the selected factors or
hyperparameters will retain their performance in future data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FACTOR_LABELS = {
    "mr_15d": "15-day mean reversion",
    "mr_17d": "17-day mean reversion",
    "vol_change": "Change in 60-day volatility",
}


def information_coefficient(signal: pd.Series, target: pd.Series) -> float:
    """Time-series IC: Spearman correlation with the next-day return."""
    aligned = pd.concat([signal.rename("signal"), target.rename("target")], axis=1).dropna()
    return float(spearmanr(aligned["signal"], aligned["target"]).statistic)


def download_spy() -> pd.DataFrame:
    """Download all available adjusted daily SPY OHLCV observations."""
    prices = yf.download(
        "SPY",
        period="max",
        interval="1d",
        auto_adjust=True,
        actions=False,
        progress=False,
    )
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    required = {"Open", "High", "Low", "Close", "Volume"}
    if prices.empty or not required.issubset(prices.columns):
        raise RuntimeError("Yahoo Finance did not return complete daily SPY OHLCV data.")
    return prices.sort_index()


def build_dataset(prices: pd.DataFrame) -> pd.DataFrame:
    close = prices["Close"].astype(float)
    daily_return = close.pct_change()
    realised_vol_60d = daily_return.rolling(60).std() * np.sqrt(252)

    return pd.DataFrame(
        {
            # Similar lookbacks intentionally make these two factors redundant.
            "mr_15d": -close.pct_change(15),
            "mr_17d": -close.pct_change(17),
            # A volatility-state change, not a directional price forecast.
            "vol_change": realised_vol_60d.pct_change(5),
            "next_day_return": close.pct_change().shift(-1),
        }
    ).dropna()


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    model_kind: str,
) -> pd.Series:
    if model_kind == "linear":
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    elif model_kind == "nonlinear":
        model = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=7,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=7,
        )
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")

    model.fit(train[features], train["next_day_return"])
    return pd.Series(model.predict(test[features]), index=test.index)


def evaluate(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_at = int(0.70 * len(dataset))
    train, test = dataset.iloc[:split_at], dataset.iloc[split_at:]
    target = test["next_day_return"]

    rows: list[dict[str, object]] = []
    for factor in FACTOR_LABELS:
        rows.append(
            {
                "category": "Standalone factor",
                "signal": FACTOR_LABELS[factor],
                "model": "None",
                "test_ic": information_coefficient(test[factor], target),
            }
        )

    books = {
        "Mean reversion 1 only": ["mr_15d"],
        "Both mean-reversion factors": ["mr_15d", "mr_17d"],
        "Mean reversion 1 + volatility": ["mr_15d", "vol_change"],
    }
    for model_kind in ("linear", "nonlinear"):
        for label, features in books.items():
            prediction = fit_predict(train, test, features, model_kind)
            rows.append(
                {
                    "category": "Combined signal",
                    "signal": label,
                    "model": "Ridge" if model_kind == "linear" else "Gradient boosting",
                    "test_ic": information_coefficient(prediction, target),
                }
            )

    return pd.DataFrame(rows), train, test


def plot_results(results: pd.DataFrame, destination: Path) -> None:
    standalone = results[results["category"] == "Standalone factor"]
    combined = results[results["category"] == "Combined signal"]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))
    axes[0].bar(standalone["signal"], standalone["test_ic"], color=["#355C7D", "#4F81A8", "#B07C3E"])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Standalone factor IC")
    axes[0].set_ylabel("Test IC (Spearman)")
    axes[0].tick_params(axis="x", rotation=18)

    labels = list(combined["signal"].drop_duplicates())
    x = np.arange(len(labels))
    width = 0.36
    for offset, model, colour in [(-width / 2, "Ridge", "#8C8C8C"), (width / 2, "Gradient boosting", "#3A7D44")]:
        values = combined.set_index(["signal", "model"]).loc[[(label, model) for label in labels], "test_ic"]
        axes[1].bar(x + offset, values.to_numpy(), width, label=model, color=colour)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_title("IC after fitting the factor book")
    axes[1].legend(frameon=False)

    fig.suptitle("A weak standalone factor can add value through a nonlinear interaction", y=1.02)
    fig.tight_layout()
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/thesis_examples/spy_ic_complementarity"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prices = download_spy()
    dataset = build_dataset(prices)
    results, train, test = evaluate(dataset)

    prices.to_csv(args.output_dir / "spy_daily_ohlcv.csv")
    results.to_csv(args.output_dir / "ic_results.csv", index=False)
    plot_results(results, args.output_dir / "ic_complementarity.png")

    print(f"Yahoo Finance observations: {len(prices):,} ({prices.index.min().date()} to {prices.index.max().date()})")
    print(f"Training sample: {train.index.min().date()} to {train.index.max().date()} ({len(train):,} rows)")
    print(f"Test sample:     {test.index.min().date()} to {test.index.max().date()} ({len(test):,} rows)")
    print(results.to_string(index=False, formatters={"test_ic": "{:.4f}".format}))
    print(f"\nOutputs written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
