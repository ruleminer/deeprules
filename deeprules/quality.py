"""Contains utility functions for assessing rules quality.

Those a mostly some hacky ways of squeezing a bit better performance out of
`decision-rules <https://github.com/ruleminer/decision-rules>`_ functions and methods.
"""
import numpy as np
from decision_rules.conditions import AbstractCondition
from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import LogicOperators
from decision_rules.core.coverage import Coverage
from decision_rules.core.rule import AbstractRule
from decision_rules.regression import RegressionRule

from deeprules.cache import ConditionsCoverageCache
from deeprules.conditions_induction.weights import get_condition_weight


def _calculate_covered_mask(
    condition: CompoundCondition, X: np.ndarray, cache: ConditionsCoverageCache
):
    if isinstance(condition, CompoundCondition):
        if len(condition.subconditions) == 0:
            return np.ones(X.shape[0], dtype=bool)
        tmp = [
            _calculate_covered_mask(subcondition, X, cache)
            for subcondition in condition.subconditions
        ]
        arrays_sum = np.sum(
            np.array(tmp),
            axis=0,
        )
        if condition.logic_operator == LogicOperators.CONJUNCTION:
            return arrays_sum == len(condition.subconditions)
        elif condition.logic_operator == LogicOperators.ALTERNATIVE:
            return arrays_sum > 0
        else:
            raise ValueError(
                f"Unknown logic operator: {condition.logic_operator}")
    else:
        return cache.get_or_calculate(condition, X, save_to_cache=True)


def _update_conclusion_for_regression_rule(
    rule: RegressionRule, y: np.ndarray, covered_mask: np.ndarray
):
    covered_y: np.ndarray = y[covered_mask]
    if covered_y.shape[0] == 0:
        rule.conclusion.train_covered_y_std = np.nan
        rule.conclusion.train_covered_y_mean = np.nan
        rule.conclusion.train_covered_y_min = np.nan
        rule.conclusion.train_covered_y_max = np.nan
    else:
        y_mean: float = np.mean(covered_y)
        rule.conclusion.train_covered_y_std = np.sqrt(
            (np.sum(np.square(covered_y)) /
             covered_y.shape[0]) - (y_mean * y_mean)
        )
        rule.conclusion.train_covered_y_mean = y_mean
        rule.conclusion.train_covered_y_min = np.min(covered_y)
        rule.conclusion.train_covered_y_max = np.max(covered_y)
    if not rule.conclusion.fixed:
        rule.conclusion.value = rule.conclusion.train_covered_y_mean
        rule.conclusion.calculate_low_high()


def calculate_covering_info(
    rule: AbstractRule,
    X: np.ndarray,
    y: np.ndarray,
    measure: callable,
    cache: ConditionsCoverageCache,
) -> tuple[set[int], Coverage, float]:
    covered_mask = _calculate_covered_mask(rule.premise, X, cache)
    if isinstance(rule, RegressionRule):
        _update_conclusion_for_regression_rule(rule, y, covered_mask)
    positive_mask: np.ndarray = rule.conclusion.positives_mask(y)
    positive_covered_mask: np.ndarray = positive_mask & covered_mask
    negative_covered_mask: np.ndarray = covered_mask ^ positive_covered_mask
    P: int = y[positive_mask].shape[0]
    N: int = y.shape[0] - P
    p = np.count_nonzero(positive_covered_mask)
    n = np.count_nonzero(negative_covered_mask)

    covered: set[int] = set(np.where(covered_mask == 1)[0])
    coverage: Coverage = Coverage(p, n, P, N)
    quality: float = measure(coverage)
    return covered, coverage, quality


def is_condition_better_than_current_best(
    new_c: tuple[AbstractCondition, set[int], float],
    current_c_best: tuple[AbstractCondition, set[int], float],
) -> bool:
    c, c_covered, c_quality = new_c
    c_best, c_best_covered, c_best_quality = current_c_best

    if current_c_best[0] is None:
        return True

    # compare by quality
    found_better: bool = c_quality > c_best_quality
    if not found_better and c_quality == c_best_quality:
        # compare by number of covered examples
        found_better = len(c_covered) > len(c_best_covered)
        if not found_better:
            # prefer conditions with smaller number of attributes
            found_better = len(c.attributes) < len(c_best.attributes)
            if not found_better:
                # prefer conditions with higher weight
                found_better = get_condition_weight(
                    c) > get_condition_weight(c_best)
    return found_better
