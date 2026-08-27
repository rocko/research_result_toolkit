"""Export standardized analysis tables."""

from pathlib import Path

import pandas as pd


def write_csv(table: pd.DataFrame, path: str | Path) -> Path:
    """Write a result table as CSV and return the output path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(target, index=False)
    return target


def write_latex(
    table: pd.DataFrame,
    path: str | Path,
    *,
    float_precision: int = 4,
) -> Path:
    """Write a simple LaTeX tabular fragment without optional dependencies."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    def format_value(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.{float_precision}f}"
        return str(value)

    line_end = " \\\\"
    header = " & ".join(str(column) for column in table.columns) + line_end
    rows = [
        " & ".join(format_value(value) for value in row) + line_end
        for row in table.itertuples(index=False, name=None)
    ]

    content = "\n".join(
        [
            "\\begin{tabular}{" + "l" * len(table.columns) + "}",
            header,
            "\\hline",
            *rows,
            "\\end{tabular}",
            "",
        ]
    )
    target.write_text(content, encoding="utf-8")
    return target
