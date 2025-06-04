from typing import Optional

import numpy as np
import pandas as pd
from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import LogicOperators
from decision_rules.core.condition import AbstractCondition
from decision_rules.core.coverage import Coverage
from decision_rules.regression import RegressionConclusion
from decision_rules.regression import RegressionRule
from decision_rules.regression import RegressionRuleSet

from deeprules import _helpers
from deeprules._induction import RuleInducersMixin
from deeprules.cache import ConditionsCoverageCache
from deeprules.conditions_induction import ConditionsGenerator
from deeprules.quality import calculate_covering_info
from deeprules.quality import is_condition_better_than_current_best
from deeprules.regression.dnf._params import AlgorithmParams


class RuleInducer(RuleInducersMixin):
    """Trains a regression ruleset based on given data"""

    def __init__(self, params: AlgorithmParams):
        super().__init__()
        self.params: AlgorithmParams = params
        self.cache: ConditionsCoverageCache = ConditionsCoverageCache()

    def induce_ruleset(self, X: pd.DataFrame, y: pd.Series) -> RegressionRuleSet:
        """Induces a regression ruleset based on given data

        Args:
            X (pd.DataFrame): data
            y (pd.Series): labels

        Returns:
            RegressionRuleSet: ruleset
        """
        rules: list[RegressionRule] = []

        X_np: np.ndarray = X.to_numpy()
        y_np: np.ndarray = y.to_numpy()
        uncovered: set[int] = set(range(len(y_np)))
        PN: int = len(uncovered)
        carry_on: bool = True
        while carry_on:
            rule = RegressionRule(
                premise=CompoundCondition(
                    subconditions=[], logic_operator=LogicOperators.CONJUNCTION
                ),
                conclusion=RegressionConclusion(
                    column_name=y.name,
                    value=np.nan,
                    low=float("-inf"),
                    high=float("inf"),
                ),
                column_names=list(X.columns),
            )
            rule, carry_on = self._grow(rule, X, X_np, y_np, uncovered)
            if carry_on:
                if self.params["enable_pruning"]:
                    self._prune(rule, uncovered, X_np, y_np)

                covered: set[int] = set(
                    np.where(rule.premise.covered_mask(X_np) == 1)[0]
                )
                new_uncovered: set[int] = uncovered.difference(covered)
                if len(new_uncovered) == len(uncovered):
                    carry_on = False
                if len(new_uncovered) <= self.params["max_uncovered_fraction"] * PN:
                    carry_on = False

                uncovered = new_uncovered

                rules.append(rule)
        ruleset = RegressionRuleSet(rules=rules)
        ruleset.update(X, y, self.params["voting_measure"])
        return ruleset

    def _grow(
        self,
        rule: RegressionRule,
        X_df: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
        uncovered: set[int],
    ) -> tuple[RegressionRule, bool]:
        all_intermediate_rules: list[tuple[RegressionRule, float]] = []
        while True:
            disjunction, intermediate_rules = self._grow_disjunction(
                rule, X_df, X, y, uncovered
            )
            all_intermediate_rules += intermediate_rules
            if (
                len(
                    rule.premise.subconditions) > self.params["max_layers_count"]
                or len(disjunction.subconditions) == 0
            ):
                break
            rule = RegressionRule(
                premise=CompoundCondition(
                    subconditions=[s for s in rule.premise.subconditions]
                    + [disjunction]
                ),
                conclusion=rule.conclusion,
                column_names=rule.column_names,
            )
        if len(all_intermediate_rules) == 0:
            return None, False
        # find the best rule from all intermediate rules (select best candidate)
        rule: RegressionRule = sorted(
            all_intermediate_rules,
            key=lambda e: self.params["select_best_candidate_measure"](
                e[0].coverage),
            reverse=True,
        )[0][0]
        return rule, len(rule.premise.subconditions) > 0

    def _grow_disjunction(
        self,
        rule: RegressionRule,
        X_df: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
        uncovered: set[int],
    ) -> tuple[CompoundCondition, list[tuple[RegressionRule, float]]]:
        disjunction: CompoundCondition = CompoundCondition(
            subconditions=[], logic_operator=LogicOperators.ALTERNATIVE
        )
        intermediate_rules: list[tuple[float, RegressionRule]] = []

        _, cov_best, _ = calculate_covering_info(
            rule, X, y, self.params["quality_measure"], self.cache
        )

        while True:
            c_best, q_best, cov_best = self._find_best_candidate_for_disjunction(
                rule, cov_best, disjunction, X_df, X, y, uncovered
            )
            if c_best is None:
                break
            disjunction.subconditions.append(c_best)
            intermediate_rule = RegressionRule(
                premise=CompoundCondition(
                    subconditions=[s for s in rule.premise.subconditions],
                    logic_operator=LogicOperators.CONJUNCTION,
                ),
                conclusion=rule.conclusion,
                column_names=rule.column_names,
            )
            intermediate_disjunction = CompoundCondition(
                subconditions=[s for s in disjunction.subconditions],
                logic_operator=LogicOperators.ALTERNATIVE,
            )
            intermediate_rule.coverage = cov_best
            intermediate_rule.premise.subconditions.append(
                intermediate_disjunction)
            intermediate_rules.append((intermediate_rule, q_best))

            if len(disjunction.subconditions) >= self.params["max_disjunction_length"]:
                break
        return disjunction, intermediate_rules

    def _find_best_candidate_for_disjunction(
        self,
        rule: RegressionRule,
        cov: Coverage,
        disjunction: CompoundCondition,
        X_df: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
        uncovered: set[int],
    ) -> tuple[AbstractCondition, float]:
        q_best: float = float("-inf")
        c_best: Optional[AbstractCondition] = None
        cov_best: Optional[Coverage] = Coverage(p=0, n=0, P=0, N=0)
        cov_before: Coverage = rule.calculate_coverage(X, y)
        covered_mask: np.ndarray = rule.premise.covered_mask(X)
        covered_best: set[int] = set(np.where(covered_mask == 1)[0])
        conditions: list[AbstractCondition] = ConditionsGenerator(
            X,
            y,
            numerical_attributes_indices=_helpers.get_numerical_indexes(X_df),
            nominal_attributes_indices=_helpers.get_nominal_indexes(X_df),
            cuts_only_between_classes=True,
            enable_attributes_conditions=self.params["enable_attributes_conditions"],
            enable_negations=self.params["enable_negations"],
        ).generate_conditions(X[covered_mask], y[covered_mask])
        for c in conditions:
            disj_c = CompoundCondition(
                subconditions=disjunction.subconditions + [c],
                logic_operator=LogicOperators.ALTERNATIVE,
            )
            rc = RegressionRule(
                premise=CompoundCondition(
                    subconditions=rule.premise.subconditions + [disj_c],
                    logic_operator=LogicOperators.CONJUNCTION,
                ),
                conclusion=RegressionConclusion(
                    value=rule.conclusion.value,
                    column_name=rule.conclusion.column_name,
                    low=rule.conclusion.low,
                    high=rule.conclusion.high,
                ),
                column_names=rule.column_names,
            )
            covered, cov, q = calculate_covering_info(
                rc, X, y, self.params["quality_measure"], self.cache
            )
            if is_condition_better_than_current_best(
                (c, covered, q), (c_best, covered_best, q_best)
            ):
                new_covered_examples = len(covered.intersection(uncovered))
                if self._check_candidate(new_covered_examples, len(covered), uncovered):
                    q_best = q
                    c_best = c
                    cov_best = cov
                    covered_best = covered

        if cov_before.p + cov_before.n == cov_best.p + cov_best.n:
            return None, np.nan, np.nan
        return c_best, q_best, cov_best

    def _check_candidate(
        self, new_covered_examples: int, covered_count: int, uncovered: set[int]
    ) -> bool:
        adjusted_min_cov = min(self.params["min_cov"], len(uncovered))
        return (new_covered_examples >= adjusted_min_cov) and (
            covered_count >= self.params["min_cov"]
        )

    def _prune(
        self,
        rule: RegressionRule,
        uncovered: set[int],
        X: np.ndarray,
        y: np.ndarray,
    ):
        if len(_helpers.get_bottom_level_conditions(rule.premise)) == 1:
            return
        while True:
            c_to_remove: AbstractCondition = None
            c_to_remove_parent: CompoundCondition = None

            _, _, q_best = calculate_covering_info(
                rule, X, y, self.params["pruning_measure"], self.cache
            )
            for c, c_parent in _helpers.get_bottom_level_conditions(rule.premise):
                old_parent_subconditions: list[AbstractCondition] = [
                    e for e in c_parent.subconditions
                ]
                c_parent.subconditions.remove(c)
                old_conclusion = rule.conclusion
                rule.conclusion = RegressionConclusion(
                    value=old_conclusion.value,
                    column_name=old_conclusion.column_name,
                    low=old_conclusion.low,
                    high=old_conclusion.high,
                )
                covered, _, q_pruned = calculate_covering_info(
                    rule, X, y, self.params["pruning_measure"], self.cache
                )
                new_covered_examples: set[int] = len(
                    covered.intersection(uncovered))
                if q_pruned >= q_best and self._check_candidate(
                    new_covered_examples, len(covered), uncovered
                ):
                    q_best = q_pruned
                    c_to_remove, c_to_remove_parent = c, c_parent
                # restore parent condition
                rule.conclusion = old_conclusion
                c_parent.subconditions = old_parent_subconditions
            if c_to_remove is not None:
                c_to_remove_parent.subconditions.remove(c_to_remove)
            else:
                break
            if len(_helpers.get_bottom_level_conditions(rule.premise)) == 1:
                break

            for i, c in enumerate(rule.premise.subconditions):
                if isinstance(c, CompoundCondition) and len(c.subconditions) == 1:
                    rule.premise.subconditions[i] = c.subconditions[0]
            # remove empty disjunctions and reduce ones with single element
            to_remove: list[AbstractCondition] = []
            for i, subcondition in enumerate(rule.premise.subconditions):
                if (
                    isinstance(subcondition, CompoundCondition)
                    and len(subcondition.subconditions) == 0
                ):
                    to_remove.append(subcondition)
            for condition_to_remove in to_remove:
                rule.premise.subconditions.remove(condition_to_remove)
