"""Small utilities for standardized analysis of repeated experiment results."""

from .evaluation import load_evaluation, run_evaluation
from .metrics import summarize
from .schema import REQUIRED_COLUMNS, load_results, validate_results
from .statistics import (
    ComparisonResult,
    adjust_p_values,
    bonferroni_adjust,
    compare,
    holm_adjust,
    pairwise_compare,
    suggest_test,
)

__all__ = [
    "ComparisonResult",
    "REQUIRED_COLUMNS",
    "adjust_p_values",
    "bonferroni_adjust",
    "compare",
    "holm_adjust",
    "load_evaluation",
    "load_results",
    "pairwise_compare",
    "run_evaluation",
    "suggest_test",
    "summarize",
    "validate_results",
]
