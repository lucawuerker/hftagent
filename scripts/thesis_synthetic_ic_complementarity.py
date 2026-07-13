"""Synthetic illustration of nonlinear factor complementarity.

The data-generating process contains two redundant directional factors and one
volatility-state factor.  Volatility has no standalone directional effect, but
modulates the strength of the first factor through an explicit interaction:

    forward_return = beta * latent_alpha
                   + gamma * latent_alpha * volatility_state
                   + noise.

Consequently, a nonlinear model can extract information from volatility that a
linear additive model cannot.  IC is Pearson correlation throughout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FACTOR_LABELS = {
    "alpha_1": "Mean reversion 1",
    "alpha_2": "Mean reversion 2",
    "volatility": "Volatility state",
}


def information_coefficient(signal: pd.Series, target: pd.Series) -> float:
    """Pearson correlation between a signal and the forward return."""
    aligned = pd.concat([signal.rename("signal"), target.rename("target")], axis=1).dropna()
    return float(aligned["signal"].corr(aligned["target"], method="pearson"))


def simulate(n_observations: int = 10_000, seed: int = 7) -> pd.DataFrame:
    """Generate a reproducible factor panel with a known nonlinear interaction."""
    rng = np.random.default_rng(seed)

    latent_alpha = rng.standard_normal(n_observations)
    volatility = rng.lognormal(mean=0.0, sigma=0.45, size=n_observations)
    volatility_state = (volatility - volatility.mean()) / volatility.std()

    # Both candidate alphas measure nearly the same latent predictive component.
    alpha_1 = latent_alpha + 0.20 * rng.standard_normal(n_observations)
    alpha_2 = latent_alpha + 0.20 * rng.standard_normal(n_observations)

    # Volatility is independent of direction and therefore has approximately zero
    # standalone IC.  Its value is instead conditional: it changes the strength of
    # the latent alpha through the alpha × volatility interaction.
    forward_return = (
        0.10 * latent_alpha
        + 0.28 * latent_alpha * volatility_state
        + rng.standard_normal(n_observations)
    )

    return pd.DataFrame(
        {
            "alpha_1": alpha_1,
            "alpha_2": alpha_2,
            "volatility": volatility_state,
            "forward_return": forward_return,
        },
        index=pd.RangeIndex(n_observations, name="t"),
    )


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    model_kind: str,
) -> pd.Series:
    if model_kind == "linear":
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    elif model_kind == "nonlinear":
        model = LGBMRegressor(
            objective="regression",
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=7,
            max_depth=3,
            min_child_samples=100,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            random_state=7,
            n_jobs=1,
            verbosity=-1,
        )
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")

    model.fit(train[features], train["forward_return"])
    return pd.Series(model.predict(test[features]), index=test.index)


def evaluate(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_at = int(0.70 * len(dataset))
    train, test = dataset.iloc[:split_at], dataset.iloc[split_at:]
    target = test["forward_return"]

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
        "Mean reversion 1 only": ["alpha_1"],
        "Both mean-reversion factors": ["alpha_1", "alpha_2"],
        "Mean reversion 1 + volatility": ["alpha_1", "volatility"],
    }
    for model_kind in ("linear", "nonlinear"):
        for label, features in books.items():
            prediction = fit_predict(train, test, features, model_kind)
            rows.append(
                {
                    "category": "Combined signal",
                    "signal": label,
                    "model": "Ridge" if model_kind == "linear" else "LightGBM",
                    "test_ic": information_coefficient(prediction, target),
                }
            )

    # This is shown as a transparent reference for the interaction embedded in the
    # DGP, not as another fitted model.
    oracle = test["alpha_1"] * test["volatility"]
    rows.append(
        {
            "category": "Diagnostic",
            "signal": "Explicit alpha × volatility interaction",
            "model": "Known interaction",
            "test_ic": information_coefficient(oracle, target),
        }
    )
    return pd.DataFrame(rows), train, test


def plot_results(results: pd.DataFrame, output_dir: Path) -> None:
    standalone = results[results["category"] == "Standalone factor"]
    combined = results[results["category"] == "Combined signal"]

    # 1. Standalone Factor Plot
    fig1, ax1 = plt.subplots(figsize=(6.0, 4.5))
    ax1.bar(
        standalone["signal"],
        standalone["test_ic"],
        color=["#355C7D", "#4F81A8", "#B07C3E"],
    )
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_title("Standalone Factor IC")
    ax1.set_ylabel("Test IC")
    ax1.tick_params(axis="x", rotation=18)
    fig1.tight_layout()
    fig1.savefig(output_dir / "ic_standalone.png", dpi=220, bbox_inches="tight")
    plt.close(fig1)

    # 2. Combined Signal Plot
    fig2, ax2 = plt.subplots(figsize=(6.0, 4.5))
    labels = list(combined["signal"].drop_duplicates())
    x = np.arange(len(labels))
    width = 0.36
    for offset, model, colour in [
        (-width / 2, "Ridge", "#8C8C8C"),
        (width / 2, "LightGBM", "#3A7D44"),
    ]:
        indexed = combined.set_index(["signal", "model"])
        values = indexed.loc[[(label, model) for label in labels], "test_ic"]
        ax2.bar(x + offset, values.to_numpy(), width, label=model, color=colour)
    
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x, labels, rotation=18, ha="right")
    ax2.set_title("IC After Fitting the Factor Book")
    ax2.legend(frameon=False)
    fig2.tight_layout()
    fig2.savefig(output_dir / "ic_combined.png", dpi=220, bbox_inches="tight")
    plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-observations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/thesis_examples/synthetic_ic_complementarity"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = simulate(args.n_observations, args.seed)
    results, train, test = evaluate(dataset)
    dataset.to_csv(args.output_dir / "synthetic_factor_data.csv")
    results.to_csv(args.output_dir / "ic_results.csv", index=False)
    
    # Pass the directory directly instead of a single filename
    plot_results(results, args.output_dir)

    print(f"Training observations: {len(train):,}")
    print(f"Test observations:     {len(test):,}")
    print(results.to_string(index=False, formatters={"test_ic": "{:.4f}".format}))
    print(f"\nOutputs written to {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()
