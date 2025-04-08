import math
from datetime import timedelta
from logging import Logger, getLogger
from typing import Callable, Optional, TypedDict

import numpy as np
import pandas as pd
from decision_rules.classification import (ClassificationConclusion,
                                           ClassificationRuleSet)
from sklearn.metrics import balanced_accuracy_score

from deeprules._induction import RuleInductionTimes
from deeprules._timing import PerformanceTimer
from deeprules.classification.mixed._helpers import (RuleSetGrower,
                                                     RuleSetGrowingContext,
                                                     RuleSetGrowingResult,
                                                     RuleSetGrowingResults)
from deeprules.classification.mixed._params import AlgorithmParams


class _HistoryEntry(TypedDict):
    pred_quality: float
    fraction_uncovered: float
    induction_times: RuleInductionTimes


class GreedyRuleInducer:
    """Trains a classification ruleset based on given data using greedy search"""

    def __init__(self, params: AlgorithmParams):
        super().__init__()
        self.params: AlgorithmParams = params
        self.prediction_quality_metric: Callable[[np.ndarray, np.ndarray], float] = (
            balanced_accuracy_score
        )
        self.logger: Logger = getLogger(self.__class__.__name__)
        self.logger.disabled = True
        self.result: RuleSetGrowingResult = None

        self.history: list[_HistoryEntry] = []
        self.induction_times: RuleInductionTimes = RuleInductionTimes()

    def induce_ruleset(self, X: pd.DataFrame, y: pd.Series) -> ClassificationRuleSet:
        """Induces a classification ruleset based on given data

        Args:
            X (pd.DataFrame): data
            y (pd.Series): labels

        Returns:
            ClassificationRuleSet: ruleset
        """
        timer = PerformanceTimer()
        timer.__enter__()
        majority_class: str = None  # classes[np.argmax(classes_sizes)]
        # Here we have to induces rules one by one for each class successively
        self.result = RuleSetGrowingResult(
            type=None,
            ruleset=ClassificationRuleSet(rules=[]),
            prediction_quality=float("-inf"),
            context=RuleSetGrowingContext.create_initial_context(
                X, y, self.params, self.prediction_quality_metric
            ),
        )
        self.result.ruleset.default_conclusion = ClassificationConclusion(
            value=majority_class, column_name=y.name
        )
        self.save_history_entry(grower=None)

        # Start with the minority class to achieve a working single rule classifier
        # the default rule is always a majority class so that way we can predict
        classes, classes_sizes = (
            self.result.context.classes,
            self.result.context.classes_sizes,
        )
        sorted_indices: np.ndarray = np.argsort(classes_sizes)[::-1]
        classes, classes_sizes = (
            self.result.context.classes[sorted_indices],
            self.result.context.classes_sizes[sorted_indices],
        )
        last_pred_quality: float = float("-inf")
        carry_on: bool = True

        while carry_on:
            for class_value, class_size in zip(classes, classes_sizes):
                self.logger.info("Inducing rules for class %s", class_value)

                grower = RuleSetGrower(self.result)
                results: RuleSetGrowingResults = grower.grow_single_rule(
                    class_value, class_size
                )
                if results.carry_on is None:
                    self.result.context.finished_classes = (
                        results.dnf.context.finished_classes
                    )
                    continue
                result: RuleSetGrowingResult = self.select_best_rule(
                    results, class_value, last_pred_quality
                )
                if len(result.context.finished_classes["cnf"]) == len(classes) and len(
                    result.context.finished_classes["dnf"]
                ) == len(classes):
                    carry_on = False
                    break
                # check if both algorithms have finished
                if result.rule is None:
                    self.result.context.finished_classes["cnf"].add(class_value)
                    self.result.context.finished_classes["dnf"].add(class_value)
                    continue

                last_pred_quality = result.prediction_quality
                new_uncovered: set[int] = result.context.class_uncovered[class_value]
                last_uncovered: set[int] = self.result.context.class_uncovered[
                    class_value
                ]
                if (
                    len(new_uncovered)
                    <= self.params["max_uncovered_fraction"] * class_size
                ):
                    # this conditions finish induction for both cnf and dnf for class
                    result.context.finished_classes["dnf"].add(class_value)
                    result.context.finished_classes["cnf"].add(class_value)

                if len(new_uncovered) == len(last_uncovered):
                    result.context.finished_classes[result.type].add(class_value)
                else:
                    self.result = result
                    self.save_history_entry(grower)

        self.ruleset = ClassificationRuleSet(rules=self.result.ruleset.rules)
        self.ruleset.update(X, y, self.params["voting_measure"])

        timer.__exit__()
        self._setup_induction_times(timer.timedelta)
        return self.ruleset

    def select_best_rule(
        self, results: RuleSetGrowingResults, class_value: str, last_pred_quality: float
    ) -> RuleSetGrowingResult:
        if results.dnf is None and results.cnf is None:
            return None

        # Now select the best rule and add it to the ruleset
        dnf_rule, cnf_rule = results.dnf.rule, results.cnf.rule
        dnf_pred_quality, cnf_pred_quality = (
            results.dnf.prediction_quality,
            results.cnf.prediction_quality,
        )

        result: RuleSetGrowingResult = None
        quality: float = float("-inf")

        if dnf_rule is None:
            results.dnf.context.finished_classes["dnf"].add(class_value)
        if cnf_rule is None:
            results.cnf.context.finished_classes["cnf"].add(class_value)

        # If only one rule is available, select it but only if it improves the quality
        if dnf_rule is None or cnf_rule is None:
            result: RuleSetGrowingResult = (
                results.dnf if dnf_rule is not None else results.cnf
            )
            if result.prediction_quality > last_pred_quality:
                self.logger.info(
                    'Selected rule: "%s" with q = %f',
                    str(result.rule),
                    quality,
                )
            return result
        if (
            dnf_pred_quality > cnf_pred_quality
            and dnf_rule is not None
            and cnf_rule is not None
        ):
            self.logger.info(
                'Selected DNF rule: "%s" with q = %f',
                str(cnf_rule),
                dnf_pred_quality,
            )
            return results.dnf
        if (
            cnf_pred_quality > dnf_pred_quality
            and cnf_rule is not None
            and dnf_rule is not None
        ):
            self.logger.info(
                'Selected CNF rule: "%s" with q = %f',
                str(cnf_rule),
                cnf_pred_quality,
            )
            return results.cnf
        # Case when both rules have the same predictive quality, select one with better p
        if math.isclose(cnf_pred_quality, dnf_pred_quality):
            if cnf_rule.coverage.p > dnf_rule.coverage.p:
                self.logger.info(
                    'Selected CNF rule: "%s" with q = %f',
                    str(cnf_rule),
                    cnf_pred_quality,
                )
                return results.cnf
            else:
                self.logger.info(
                    'Selected DNF rule: "%s" with q = %f',
                    str(cnf_rule),
                    dnf_pred_quality,
                )
                return results.dnf

        return result

    def save_history_entry(self, grower: Optional[RuleSetGrower]):
        if grower is not None:
            times: RuleInductionTimes = (
                grower.dnf_inducer.induction_times
                if grower.dnf_inducer.induction_times.total_training_time
                > grower.cnf_inducer.induction_times.total_training_time
                else grower.cnf_inducer.induction_times
            )
        else:
            times = RuleInductionTimes()
        self.history.append(
            _HistoryEntry(
                pred_quality=self.result.prediction_quality,
                fraction_uncovered=(
                    sum([len(s) for s in self.result.context.class_uncovered.values()])
                    / self.result.context.X.shape[0]
                ),
                induction_times=times,
            )
        )

    def _setup_induction_times(self, total_training_time: timedelta):
        # sum all times from all history entries
        self.induction_times: RuleInductionTimes = sum(
            [entry["induction_times"] for entry in self.history]
        )
        # restore real total training time
        self.induction_times.total_training_time = total_training_time
