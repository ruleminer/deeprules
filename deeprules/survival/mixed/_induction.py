import math
from datetime import timedelta
from logging import Logger, getLogger
from typing import Callable, Optional, TypedDict

import numpy as np
import pandas as pd
from decision_rules.survival import (KaplanMeierEstimator, SurvivalConclusion,
                                     SurvivalRuleSet)

from deeprules._induction import RuleInductionTimes
from deeprules._timing import PerformanceTimer
from deeprules.survival.mixed._helpers import (RuleSetGrower,
                                               RuleSetGrowingContext,
                                               RuleSetGrowingResult,
                                               RuleSetGrowingResults)
from deeprules.survival.mixed._params import AlgorithmParams


class _HistoryEntry(TypedDict):
    pred_quality: float
    fraction_uncovered: float
    induction_times: RuleInductionTimes


class  GreedyRuleInducer:
    """Trains a survival ruleset based on given data using greedy search"""

    def __init__(self, params: AlgorithmParams):
        super().__init__()
        self.params: AlgorithmParams = params
        self.prediction_quality_metric: Callable[
            [SurvivalRuleSet, np.ndarray, np.ndarray], float
        ] = lambda ruleset, X, y: ruleset.integrated_bier_score(X, y)
        self.logger: Logger = getLogger(self.__class__.__name__)
        self.logger.disabled = True
        self.result: RuleSetGrowingResult = None

        self.history: list[_HistoryEntry] = []
        self.induction_times: RuleInductionTimes = RuleInductionTimes()

    def induce_ruleset(self, X: pd.DataFrame, y: pd.Series) -> SurvivalRuleSet:
        """Induces a regression ruleset based on given data

        Args:
            X (pd.DataFrame): data
            y (pd.Series): labels

        Returns:
            RegressionRuleSet: ruleset
        """
        X = X.sort_values(by=self.params["survival_time_attr"], ascending=True)
        y = y.astype(int).astype(str)
        X_np: np.ndarray = X.to_numpy()
        y_np: np.ndarray = y.to_numpy()
        default_km = KaplanMeierEstimator().fit(
            X[self.params["survival_time_attr"]].to_numpy(), y_np
        )

        timer = PerformanceTimer()
        timer.__enter__()
        # Here we have to induces rules one by one for each class successively
        self.result = RuleSetGrowingResult(
            type=None,
            ruleset=SurvivalRuleSet(
                rules=[], survival_time_attr=self.params["survival_time_attr"]
            ),
            prediction_quality=float("-inf"),
            context=RuleSetGrowingContext.create_initial_context(
                X, y, self.params, self.prediction_quality_metric
            ),
        )

        self.result.ruleset.default_conclusion = SurvivalConclusion(
            value=default_km.median_survival_time, column_name=y.name
        )
        self.save_history_entry(grower=None)

        last_pred_quality: float = float("-inf")
        carry_on: bool = True
        PN: int = len(y)

        while carry_on:
            self.logger.info("Inducing rules")

            grower =  RuleSetGrower(self.result)
            results: RuleSetGrowingResults = grower.grow_single_rule()
            if results.carry_on is None:
                self.result.context.finished = results.dnf.context.finished
                continue
            result: RuleSetGrowingResult = self.select_best_rule(
                results, last_pred_quality
            )
            if result.context.finished["cnf"] and result.context.finished["dnf"]:
                carry_on = False
                break
            # check if both algorithms have finished
            if result.rule is None:
                self.result.context.finished["cnf"] = True
                self.result.context.finished["dnf"] = True
                continue

            last_pred_quality = result.prediction_quality
            new_uncovered: set[int] = result.context.uncovered
            last_uncovered: set[int] = self.result.context.uncovered
            if len(new_uncovered) <= self.params["max_uncovered_fraction"] * PN:
                # this conditions finish induction for both cnf and dnf for class
                result.context.finished["dnf"] = True
                result.context.finished["cnf"] = True

            if len(new_uncovered) == len(last_uncovered):
                result.context.finished[result.type] = True
            else:
                self.result = result
                self.save_history_entry(grower)

        self.ruleset = SurvivalRuleSet(
            rules=self.result.ruleset.rules,
            survival_time_attr=self.params["survival_time_attr"],
        )
        self.ruleset.update(X, y)

        timer.__exit__()
        self._setup_induction_times(timer.timedelta)
        return self.ruleset

    def select_best_rule(
        self, results: RuleSetGrowingResults, last_pred_quality: float
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
        quality: float = float("inf")  # error is better when its lower

        if dnf_rule is None:
            results.dnf.context.finished["dnf"] = True
        if cnf_rule is None:
            results.cnf.context.finished["cnf"] = True

        # If only one rule is available, select it but only if it improves the quality
        if dnf_rule is None or cnf_rule is None:
            result: RuleSetGrowingResult = (
                results.dnf if dnf_rule is not None else results.cnf
            )
            if result.prediction_quality < last_pred_quality:  # lower is better
                self.logger.info(
                    'Selected rule: "%s" with q = %f',
                    str(result.rule),
                    quality,
                )
            return result
        if (
            dnf_pred_quality < cnf_pred_quality  # lower is better
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
            cnf_pred_quality < dnf_pred_quality
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

    def save_history_entry(self, grower: Optional[ RuleSetGrower]):
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
                    len(self.result.context.uncovered) / self.result.context.X.shape[0]
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
        # restore real total training time
        self.induction_times.total_training_time = total_training_time
