"""Load and validate the common experiment-result schema."""

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"experiment", "run", "metric", "value"}


def validate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns and return a normalized copy."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_text}")

    out = df.copy()
    out["experiment"] = out["experiment"].astype(str)
    out["metric"] = out["metric"].astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="raise")

    if out[["experiment", "run", "metric", "value"]].isna().any().any():
        raise ValueError("Required result fields must not contain missing values.")

    duplicates = out.duplicated(subset=["experiment", "run", "metric"])
    if duplicates.any():
        raise ValueError(
            "Each experiment/run/metric combination must occur only once."
        )

    return out


def load_results(path: str | Path) -> pd.DataFrame:
    """Load CSV results and validate them against the common schema."""
    source = Path(path)
    if source.suffix.lower() != ".csv":
        raise ValueError("The prototype currently supports CSV input only.")

    return validate_results(pd.read_csv(source))
