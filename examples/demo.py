"""Run the complete toolkit workflow on a small example dataset."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stats_toolkit import (
    compare,
    load_results,
    pairwise_compare,
    suggest_test,
    summarize,
)
from stats_toolkit.plotting import plot_metric
from stats_toolkit.reporting import write_csv, write_latex


def main() -> None:
    results = load_results(ROOT / "examples" / "example_results.csv")
    summary = summarize(results)

    print("Summary")
    print(summary.to_string(index=False))

    f1_sample_size = int(
        results.loc[
            (results["experiment"] == "baseline") & (results["metric"] == "f1")
        ].shape[0]
    )
    guidance = suggest_test(
        sample_size=f1_sample_size,
        paired=True,
        normality_assumed=None,
    )
    print("\nSuggested comparison")
    for key, value in guidance.items():
        print(f"{key}: {value}")

    comparison = compare(
        results,
        "baseline",
        "method_a",
        metric="f1",
        test="wilcoxon",
        paired=True,
    )
    print("\nComparison")
    print(comparison)

    experiments = sorted(results["experiment"].unique().tolist())
    pairwise = pairwise_compare(
        results,
        experiments,
        metric="f1",
        test="wilcoxon",
        paired=True,
    )
    print("\nPairwise comparisons with Holm correction")
    print(pairwise.to_string(index=False))

    output_dir = ROOT / "examples" / "output"
    write_csv(summary, output_dir / "demo" / "summary.csv")
    write_csv(pairwise, output_dir / "demo" / "pairwise_f1.csv")
    write_latex(summary, output_dir / "demo" / "summary.tex")
    plot_metric(results, metric="f1", path=output_dir / "demo" / "f1.png")

    print(f"\nWrote example outputs to {output_dir}/demo")


if __name__ == "__main__":
    main()
