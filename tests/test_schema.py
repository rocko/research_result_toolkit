from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stats_toolkit import validate_results


def test_validate_results_accepts_common_schema() -> None:
    frame = pd.DataFrame(
        {
            "experiment": ["baseline"],
            "run": [1],
            "metric": ["accuracy"],
            "value": [0.8],
        }
    )

    validated = validate_results(frame)
    assert validated.loc[0, "value"] == pytest.approx(0.8)


def test_validate_results_rejects_missing_columns() -> None:
    frame = pd.DataFrame({"experiment": ["baseline"], "value": [0.8]})

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_results(frame)


def test_validate_results_rejects_duplicate_measurements() -> None:
    frame = pd.DataFrame(
        {
            "experiment": ["baseline", "baseline"],
            "run": [1, 1],
            "metric": ["accuracy", "accuracy"],
            "value": [0.8, 0.81],
        }
    )

    with pytest.raises(ValueError, match="must occur only once"):
        validate_results(frame)
