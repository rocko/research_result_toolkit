"""Execute the example declarative evaluation."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stats_toolkit.evaluation import run_evaluation


def main() -> None:
    result = run_evaluation(ROOT / "examples" / "evaluation.yaml")

    print("Summary")
    print(result["summary"].to_string(index=False))

    comparisons = result["comparisons"]
    if not comparisons.empty:
        print("\nComparisons")
        print(comparisons.to_string(index=False))

    print(f"\nWrote evaluation outputs to {result['output_dir']}")


if __name__ == "__main__":
    main()
