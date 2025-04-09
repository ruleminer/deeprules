"""Contains code for caching conditions coverage. It is done to improve performance by
avoiding recalculating each time a conditions is added or removed from a rule.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from decision_rules.conditions import AbstractCondition
from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import LogicOperators


class ConditionsCoverageCache:
    """Cache for storing conditions coverages to avoid recalculating them"""

    def __init__(self):
        self.cache: dict[str, np.ndarray] = {}

        self.hits_count: int = 0
        self.misses_count: int = 0

    def get(self, condition: AbstractCondition) -> Optional[np.ndarray]:
        # try getting condition directly
        self.hits_count += 1
        cache_key = condition.__hash__()
        coverage = self.cache.get(cache_key)
        if coverage is not None:
            return self.cache[cache_key]
        if isinstance(condition, CompoundCondition):
            # try getting condition's subconditions recursively
            subconditions_coverages = [
                self.get(subcondition) for subcondition in condition.subconditions
            ]
            if any([coverage is not None for coverage in subconditions_coverages]):
                return None
            if condition.logic_operator == LogicOperators.ALTERNATIVE:
                return np.max(subconditions_coverages, axis=0)
            elif condition.logic_operator == LogicOperators.CONJUNCTION:
                return np.min(subconditions_coverages, axis=0)
            else:
                raise ValueError(
                    f"Unknown logic operator: {condition.logic_operator}")
        self.hits_count -= 1
        self.misses_count += 1
        return None

    def get_or_calculate(
        self, condition: AbstractCondition, X: np.ndarray, save_to_cache: bool = True
    ) -> np.ndarray:
        coverage_mask: np.ndarray = self.get(condition)
        if coverage_mask is None:
            coverage_mask = condition.covered_mask(X)
        if save_to_cache:
            self.set(condition, coverage_mask)
        return coverage_mask

    def set(self, condition: AbstractCondition, value: np.ndarray):
        cache_key = condition.__hash__()
        self.cache[cache_key] = value

    def update(self, cache: ConditionsCoverageCache):
        self.cache.update(cache.cache)
