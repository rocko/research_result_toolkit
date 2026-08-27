"""Descriptive summaries for repeated experiment results."""

import numpy as np
import pandas as pd
from scipy.stats import t


def _confidence_interval(values: pd.Series, confidence: float) -> tuple[float, float]:
    """Return the confidence-interval bounds for the sample mean."""
    array = values.to_numpy(dtype=float)
    mean = float(array.mean())

    if len(array) < 2:
        return mean, mean

    standard_error = float(array.std(ddof=1) / np.sqrt(len(array)))
    critical_value = float(t.ppf((1.0 + confidence) / 2.0, len(array) - 1))
    margin = critical_value * standard_error
    return mean - margin, mean + margin


def summarize(results: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Summarize each experiment and metric across repeated runs."""
    rows: list[dict[str, object]] = []

    for (experiment, metric), group in results.groupby(["experiment", "metric"]):
        values = group["value"].astype(float)
        ci_low, ci_high = _confidence_interval(values, confidence)

        rows.append(
            {
                "experiment": experiment,
                "metric": metric,
                "n": len(values),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "median": float(values.median()),
                "min": float(values.min()),
                "max": float(values.max()),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    return pd.DataFrame(rows).sort_values(["metric", "experiment"]).reset_index(drop=True)
