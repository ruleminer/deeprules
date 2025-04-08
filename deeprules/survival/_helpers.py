import numpy as np
from decision_rules.core.coverage import Coverage
from decision_rules.survival.rule import SurvivalRule

from deeprules.cache import ConditionsCoverageCache
from deeprules.quality import _calculate_covered_mask
from deeprules.survival._kaplan_meier import PrecalculatedKaplanMeierEstimator


def calculate_coverage(
    rule: SurvivalRule,
    X: np.ndarray,
    y: np.ndarray,
    cache: ConditionsCoverageCache,
) -> Coverage:
    covered_mask: np.ndarray = _calculate_covered_mask(rule.premise, X, cache)
    uncovered_mask: np.ndarray = np.logical_not(covered_mask)
    covered_examples_indexes = np.where(covered_mask)[0]
    uncovered_examples_indexes = np.where(uncovered_mask)[0]
    survival_time: np.ndarray = X[:, rule.survival_time_attr_idx]
    rule.log_rank = PrecalculatedKaplanMeierEstimator.log_rank(
        survival_time,
        y,
        covered_examples_indexes,
        uncovered_examples_indexes,
        skip_sorting=True,
    )
    P: int = y.shape[0]
    N: int = 0
    p = np.count_nonzero(covered_mask)
    n = 0
    coverage = Coverage(p, n, P, N)
    rule.coverage = coverage
    return coverage


def calculate_covering_info(
    rule: SurvivalRule,
    X: np.ndarray,
    y: np.ndarray,
    cache: ConditionsCoverageCache,
) -> tuple[set[int], Coverage, float]:
    covered_mask = _calculate_covered_mask(rule.premise, X, cache)
    coverage = calculate_coverage(rule, X, y, cache)

    covered: set[int] = set(np.where(covered_mask == 1)[0])
    quality: float = rule.log_rank
    return covered, coverage, quality
