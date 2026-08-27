"""Standardized plots for repeated experiment metrics."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric(
    results: pd.DataFrame,
    *,
    metric: str,
    path: str | Path,
) -> Path:
    """Create a simple box plot for one metric across experiments."""
    subset = results.loc[results["metric"] == metric, ["experiment", "value"]]
    if subset.empty:
        raise ValueError(f"No values found for metric={metric!r}.")

    grouped = [
        group["value"].to_numpy(dtype=float)
        for _, group in subset.groupby("experiment", sort=True)
    ]
    labels = [name for name, _ in subset.groupby("experiment", sort=True)]

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.boxplot(grouped, tick_labels=labels)
    axis.set_title(metric)
    axis.set_ylabel("value")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=160)
    plt.close(figure)
    return target
