"""Statistical comparisons and lightweight test-selection guidance."""

from dataclasses import dataclass
from itertools import combinations
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, ttest_rel, wilcoxon


TestName = Literal[
    "paired_t",
    "wilcoxon",
    "welch_t",
    "mann_whitney",
    "permutation",
]

CorrectionName = Literal["holm", "bonferroni", "none"]


@dataclass(frozen=True)
class ComparisonResult:
    """Result of one comparison between two experiments.

    ``significant`` reports whether the raw p-value is below the configured
    alpha threshold. For families of comparisons, ``pairwise_compare()`` also
    reports corrected p-values and significance after multiple-testing
    correction.
    """

    experiment_a: str
    experiment_b: str
    metric: str
    test: str
    paired: bool
    n_a: int
    n_b: int
    statistic: float
    p_value: float
    alpha: float
    significant: bool
    mean_a: float
    mean_b: float
    mean_difference: float


def suggest_test(
    *,
    sample_size: int,
    paired: bool,
    normality_assumed: Optional[bool] = None,
) -> dict[str, str]:
    """Suggest a candidate test from explicit design assumptions.

    The recommendation is intentionally conservative and transparent. It does
    not infer whether observations are paired from the values themselves, and
    it does not treat a normality test as proof that a parametric model is valid.
    """
    if sample_size < 2:
        raise ValueError("sample_size must be at least 2.")

    sample_note = (
        f"Small sample (n={sample_size}); diagnostics have limited power and "
        "test assumptions should be checked carefully."
        if sample_size < 30
        else f"Sample size n={sample_size}; test assumptions should still be checked."
    )

    warning = (
        "Decision support only: this recommendation can be wrong for a specific "
        "study. Verify the experimental design, hypothesis, and test assumptions "
        "before making an important statistical decision."
    )

    hint = (
        "If you intend to do pairwise comparisons, consider a multiple-testing "
        "correction such as Holm or Bonferroni."
    )

    if paired:
        diagnostic = (
            "Pairing must come from the experimental design, for example matching "
            "runs that use the same split and seed. For a paired t-test, assess "
            "approximate normality of the paired differences."
        )

        if normality_assumed is True:
            test = "paired_t"
            reason = (
                "The samples are paired and approximate normality of the paired "
                "differences is assumed."
            )
            alternative = (
                "wilcoxon if normality of the paired differences is not justified"
            )
        else:
            test = "wilcoxon"
            reason = (
                "The samples are paired and normality of the paired differences "
                + ("is not assumed." if normality_assumed is False else "is unknown.")
            )
            alternative = (
                "paired_t if approximate normality of the paired differences can be justified"
            )
    else:
        diagnostic = (
            "Independence must come from the experimental design. If normality is "
            "unknown, inspect the distributions and assumptions before preferring "
            "a parametric comparison."
        )

        if normality_assumed is True:
            test = "welch_t"
            reason = (
                "The samples are independent and approximate normality is assumed; "
                "Welch's t-test does not require equal variances."
            )
            alternative = (
                "permutation or mann_whitney if parametric assumptions are unsuitable"
            )
        else:
            test = "permutation"
            reason = (
                "The samples are independent and normality "
                + ("is not assumed." if normality_assumed is False else "is unknown.")
            )
            alternative = "mann_whitney for a rank-based distributional comparison"

    return {
        "test": test,
        "reason": reason,
        "sample_size": sample_note,
        "diagnostic": diagnostic,
        "alternative": alternative,
        "warning": warning,
        "hint": hint,
    }


def _extract_values(
    results: pd.DataFrame,
    experiment: str,
    metric: str,
) -> pd.DataFrame:
    subset = results.loc[
        (results["experiment"] == experiment) & (results["metric"] == metric),
        ["run", "value"],
    ].copy()

    if subset.empty:
        raise ValueError(f"No values found for experiment={experiment!r}, metric={metric!r}.")

    return subset


def _permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
) -> tuple[float, float]:
    """Monte-Carlo permutation test for an unpaired difference in means."""
    rng = np.random.default_rng(seed)
    observed = float(x.mean() - y.mean())
    pooled = np.concatenate([x, y]).copy()
    n_x = len(x)
    extreme = 0

    for _ in range(n_permutations):
        rng.shuffle(pooled)
        difference = pooled[:n_x].mean() - pooled[n_x:].mean()
        if abs(difference) >= abs(observed):
            extreme += 1

    # Add one pseudo-count so a Monte-Carlo estimate is never reported as zero.
    p_value = (extreme + 1) / (n_permutations + 1)
    return observed, float(p_value)


def compare(
    results: pd.DataFrame,
    experiment_a: str,
    experiment_b: str,
    *,
    metric: str,
    test: TestName = "permutation",
    paired: bool = False,
    alpha: float = 0.05,
    n_permutations: int = 20_000,
    seed: int = 0,
) -> ComparisonResult:
    """Compare one metric between two experiments.

    The returned ``significant`` flag is based on the raw p-value and ``alpha``.
    If several hypotheses are tested together, use ``pairwise_compare()`` or
    another multiple-testing correction before interpreting significance.
    """
    a = _extract_values(results, experiment_a, metric)
    b = _extract_values(results, experiment_b, metric)

    if paired:
        joined = a.merge(b, on="run", suffixes=("_a", "_b"))
        if len(joined) != len(a) or len(joined) != len(b):
            raise ValueError("Paired comparisons require matching run identifiers.")
        x = joined["value_a"].to_numpy(dtype=float)
        y = joined["value_b"].to_numpy(dtype=float)
    else:
        x = a["value"].to_numpy(dtype=float)
        y = b["value"].to_numpy(dtype=float)

    if test == "paired_t":
        if not paired:
            raise ValueError("paired_t requires paired=True.")
        statistic, p_value = ttest_rel(x, y)
    elif test == "wilcoxon":
        if not paired:
            raise ValueError("wilcoxon requires paired=True.")
        statistic, p_value = wilcoxon(x, y)
    elif test == "welch_t":
        if paired:
            raise ValueError("welch_t is intended for independent samples.")
        statistic, p_value = ttest_ind(x, y, equal_var=False)
    elif test == "mann_whitney":
        if paired:
            raise ValueError("mann_whitney is intended for independent samples.")
        statistic, p_value = mannwhitneyu(x, y, alternative="two-sided")
    elif test == "permutation":
        if paired:
            raise ValueError("The prototype permutation test currently supports independent samples only.")
        statistic, p_value = _permutation_test(
            x,
            y,
            n_permutations=n_permutations,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported test: {test}")

    p_value = float(p_value)
    return ComparisonResult(
        experiment_a=experiment_a,
        experiment_b=experiment_b,
        metric=metric,
        test=test,
        paired=paired,
        n_a=len(x),
        n_b=len(y),
        statistic=float(statistic),
        p_value=p_value,
        alpha=float(alpha),
        significant=p_value < alpha,
        mean_a=float(x.mean()),
        mean_b=float(y.mean()),
        mean_difference=float(x.mean() - y.mean()),
    )


def bonferroni_adjust(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Return Bonferroni-adjusted p-values.

    Bonferroni controls the family-wise error rate by multiplying every raw
    p-value by the number of tested hypotheses, then clipping the result at 1.0.
    Equivalently, a raw p-value can be compared against alpha / m, where m is
    the number of simultaneous hypotheses.

    The method is simple and does not depend on the ordering of p-values, but it
    can be conservative when many hypotheses are tested. Holm's step-down method
    provides the same family-wise error-rate control and is generally at least as
    powerful, so Holm remains the default correction for pairwise comparisons.
    """
    values = np.asarray(p_values, dtype=float)
    if values.size == 0:
        return values.copy()

    # Mathematical counterpart: p_adjusted = min(p * m, 1), where m is the
    # number of hypotheses in the family.
    return np.minimum(values * len(values), 1.0)


def holm_adjust(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Return Holm-adjusted p-values while preserving original order.

    Holm is a sequentially rejective Bonferroni procedure for controlling the
    family-wise error rate. Raw p-values are sorted from smallest to largest.
    The smallest is multiplied by m, the next by m - 1, and so on. A cumulative
    maximum is then applied so adjusted p-values remain monotonic before they are
    restored to the original hypothesis order.

    Compared with plain Bonferroni, Holm uses progressively smaller multipliers
    and is therefore generally less conservative while providing the same type
    of family-wise error-rate control.
    """
    values = np.asarray(p_values, dtype=float)
    if values.size == 0:
        return values.copy()

    order = np.argsort(values)
    sorted_values = values[order]

    # Mathematical counterpart for sorted p-values p_(i): multiply by
    # (m - i), with i starting at zero.
    multipliers = len(values) - np.arange(len(values))
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(sorted_values * multipliers, 1.0)
    )

    # Restore adjusted p-values to the caller's original hypothesis order.
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def adjust_p_values(
    p_values: list[float] | np.ndarray,
    *,
    method: CorrectionName = "holm",
) -> np.ndarray:
    """Apply a supported multiple-testing correction to p-values.

    Supported methods are ``holm`` (step-down Bonferroni), ``bonferroni``
    (multiply every p-value by the number of hypotheses), and ``none``.
    """
    if method == "holm":
        return holm_adjust(p_values)
    if method == "bonferroni":
        return bonferroni_adjust(p_values)
    if method == "none":
        return np.asarray(p_values, dtype=float).copy()
    raise ValueError(f"Unsupported multiple-testing method: {method}")


def pairwise_compare(
    results: pd.DataFrame,
    experiments: list[str],
    *,
    metric: str,
    test: TestName = "permutation",
    paired: bool = False,
    alpha: float = 0.05,
    correction: CorrectionName = "holm",
) -> pd.DataFrame:
    """Compare all experiment pairs and correct the resulting p-values.

    Every unique pair of experiments is compared using the requested statistical
    test. ``significant_raw`` records the uncorrected alpha decision. The family
    of p-values is then corrected together and ``significant`` records the final
    decision after correction.
    """
    rows: list[dict[str, object]] = []

    for experiment_a, experiment_b in combinations(experiments, 2):
        result = compare(
            results,
            experiment_a,
            experiment_b,
            metric=metric,
            test=test,
            paired=paired,
            alpha=alpha,
        )
        row = result.__dict__.copy()
        row["significant_raw"] = row.pop("significant")
        rows.append(row)

    output = pd.DataFrame(rows)
    if output.empty:
        return output

    output["p_adjusted"] = adjust_p_values(
        output["p_value"].to_numpy(),
        method=correction,
    )
    output["significant"] = output["p_adjusted"] < alpha
    return output
