from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stats_toolkit import run_evaluation


def test_example_evaluation_runs(tmp_path: Path) -> None:
    config = tmp_path / "evaluation.yaml"
    data = tmp_path / "results.csv"

    data.write_text(
        "experiment,run,metric,value\n"
        "a,1,f1,0.80\n"
        "a,2,f1,0.82\n"
        "b,1,f1,0.84\n"
        "b,2,f1,0.85\n",
        encoding="utf-8",
    )

    config.write_text(
        "data: results.csv\n"
        "output_dir: out\n"
        "comparisons:\n"
        "  - experiment_a: a\n"
        "    experiment_b: b\n"
        "    metric: f1\n"
        "    test: wilcoxon\n"
        "    paired: true\n"
        "multiple_testing:\n"
        "  method: bonferroni\n"
        "outputs:\n"
        "  summary_csv: true\n"
        "  summary_latex: false\n"
        "  plots: []\n",
        encoding="utf-8",
    )

    result = run_evaluation(config)

    assert len(result["comparisons"]) == 1
    assert "p_adjusted" in result["comparisons"].columns
    assert (tmp_path / "out" / "summary.csv").exists()
    assert (tmp_path / "out" / "comparisons.csv").exists()
