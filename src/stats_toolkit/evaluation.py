"""Declarative evaluation runner for standardized experiment analyses."""

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .metrics import summarize
from .plotting import plot_metric
from .reporting import write_csv, write_latex
from .schema import load_results
from .statistics import adjust_p_values, compare


def load_evaluation(path: str | Path) -> dict[str, Any]:
    """Load an evaluation definition from YAML."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Evaluation configuration must contain a mapping at the top level.")

    if "data" not in config:
        raise ValueError("Evaluation configuration requires a 'data' path.")

    return config


def run_evaluation(path: str | Path) -> dict[str, object]:
    """Execute one configured evaluation and return generated result objects."""
    config_path = Path(path).resolve()
    config = load_evaluation(config_path)
    base_dir = config_path.parent

    data_path = base_dir / str(config["data"])
    results = load_results(data_path)
    summary = summarize(results)

    output_dir = base_dir / str(config.get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = config.get("outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError("'outputs' must be a mapping when provided.")

    if outputs.get("summary_csv", True):
        write_csv(summary, output_dir / "summary.csv")
    if outputs.get("summary_latex", False):
        write_latex(summary, output_dir / "summary.tex")

    plot_metrics = outputs.get("plots", [])
    if not isinstance(plot_metrics, list):
        raise ValueError("'outputs.plots' must be a list of metric names.")
    for metric in plot_metrics:
        plot_metric(results, metric=str(metric), path=output_dir / f"{metric}.png")

    comparisons_config = config.get("comparisons", [])
    if not isinstance(comparisons_config, list):
        raise ValueError("'comparisons' must be a list.")

    comparison_rows: list[dict[str, object]] = []
    for item in comparisons_config:
        if not isinstance(item, dict):
            raise ValueError("Each comparison must be a mapping.")

        result = compare(
            results,
            str(item["experiment_a"]),
            str(item["experiment_b"]),
            metric=str(item["metric"]),
            test=str(item.get("test", "permutation")),
            paired=bool(item.get("paired", False)),
            n_permutations=int(item.get("n_permutations", 20_000)),
            seed=int(item.get("seed", 0)),
        )
        comparison_rows.append(result.__dict__)

    comparisons = pd.DataFrame(comparison_rows)
    if not comparisons.empty:
        multiple_testing = config.get("multiple_testing", {})
        if not isinstance(multiple_testing, dict):
            raise ValueError("'multiple_testing' must be a mapping when provided.")

        method = str(multiple_testing.get("method", "none")).lower()
        alpha = float(multiple_testing.get("alpha", 0.05))

        comparisons["p_adjusted"] = adjust_p_values(
            comparisons["p_value"].to_numpy(),
            method=method,
        )
        comparisons["significant"] = comparisons["p_adjusted"] < alpha
        write_csv(comparisons, output_dir / "comparisons.csv")

    return {
        "config": config,
        "results": results,
        "summary": summary,
        "comparisons": comparisons,
        "output_dir": output_dir,
    }
