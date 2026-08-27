from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stats_toolkit import (
    bonferroni_adjust,
    compare,
    holm_adjust,
    pairwise_compare,
    suggest_test,
)


def _example_results() -> pd.DataFrame:
    rows = []
    for run, value in enumerate([0.80, 0.81, 0.82, 0.83, 0.84], start=1):
        rows.append({"experiment": "a", "run": run, "metric": "f1", "value": value})
    for run, value in enumerate([0.835, 0.852, 0.858, 0.879, 0.884], start=1):
        rows.append({"experiment": "b", "run": run, "metric": "f1", "value": value})
    return pd.DataFrame(rows)


def test_paired_comparison_uses_matching_runs() -> None:
    result = compare(
        _example_results(),
        "a",
        "b",
        metric="f1",
        test="paired_t",
        paired=True,
    )

    assert result.n_a == 5
    assert result.n_b == 5
    assert result.mean_difference == pytest.approx(-0.0416)
    assert result.p_value < 0.05
    assert result.alpha == pytest.approx(0.05)
    assert result.significant is True


def test_pairwise_reports_raw_and_adjusted_significance() -> None:
    output = pairwise_compare(
        _example_results(),
        ["a", "b"],
        metric="f1",
        test="paired_t",
        paired=True,
    )

    assert "significant_raw" in output.columns
    assert "p_adjusted" in output.columns
    assert "significant" in output.columns


def test_bonferroni_adjust_multiplies_by_number_of_hypotheses() -> None:
    adjusted = bonferroni_adjust([0.01, 0.04, 0.50])

    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[1] == pytest.approx(0.12)
    assert adjusted[2] == pytest.approx(1.0)


def test_holm_adjust_preserves_order_and_bounds() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])

    assert np.all(adjusted >= 0.0)
    assert np.all(adjusted <= 1.0)
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[1] == pytest.approx(0.06)
    assert adjusted[2] == pytest.approx(0.06)


def test_test_guidance_distinguishes_paired_design() -> None:
    guidance = suggest_test(
        sample_size=10,
        paired=True,
        normality_assumed=False,
    )

    assert guidance["test"] == "wilcoxon"
    assert "paired" in guidance["reason"].lower()
    assert "small sample" in guidance["sample_size"].lower()
    assert "experimental design" in guidance["diagnostic"].lower()
    assert "decision support only" in guidance["warning"].lower()


def test_test_guidance_reports_unknown_normality() -> None:
    guidance = suggest_test(
        sample_size=10,
        paired=True,
        normality_assumed=None,
    )

    assert guidance["test"] == "wilcoxon"
    assert "unknown" in guidance["reason"].lower()
    assert "paired differences" in guidance["diagnostic"].lower()


def test_test_guidance_rejects_invalid_sample_size() -> None:
    with pytest.raises(ValueError, match="sample_size"):
        suggest_test(sample_size=1, paired=False)
