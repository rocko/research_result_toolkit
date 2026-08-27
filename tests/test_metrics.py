from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stats_toolkit import summarize


def test_summarize_reports_repeated_run_statistics() -> None:
    frame = pd.DataFrame(
        {
            "experiment": ["baseline"] * 3,
            "run": [1, 2, 3],
            "metric": ["accuracy"] * 3,
            "value": [0.8, 0.9, 1.0],
        }
    )

    summary = summarize(frame)
    row = summary.iloc[0]

    assert row["n"] == 3
    assert row["mean"] == pytest.approx(0.9)
    assert row["median"] == pytest.approx(0.9)
    assert row["std"] == pytest.approx(0.1)
    assert row["ci_low"] < row["mean"] < row["ci_high"]
