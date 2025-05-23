from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from logging import Logger
from typing import Callable
from typing import Optional

import numpy as np
import pandas as pd
from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import LogicOperators
from decision_rules.core.coverage import Coverage
from decision_rules.survival import KaplanMeierEstimator
from decision_rules.survival import SurvivalConclusion
from decision_rules.survival import SurvivalRule
from decision_rules.survival import SurvivalRuleSet
from joblib import delayed
from joblib import Parallel

from deeprules.cache import ConditionsCoverageCache
from deeprules.quality import _calculate_covered_mask
from deeprules.survival._kaplan_meier import PrecalculatedKaplanMeierEstimator
from deeprules.survival.cnf._induction import RuleInducer as CNFInducer
from deeprules.survival.cnf._params import AlgorithmParams as CNFParams
from deeprules.survival.dnf._induction import RuleInducer as DNFInducer
from deeprules.survival.dnf._params import AlgorithmParams as DNFParams
from deeprules.survival.mixed._params import AlgorithmParams
from deeprules.survival.mixed._params import to_cnf_params
from deeprules.survival.mixed._params import to_dnf_params


@dataclass
class RuleChoice:
    type: str
    rule: Optional[SurvivalRule]
    pred_quality: float


@dataclass
class RuleChoices:
    dnf: RuleChoice
    cnf: RuleChoice


@dataclass
class RuleSetGrowingContext:
    params: AlgorithmParams
    X: pd.DataFrame
    y: pd.Series
    X_np: np.ndarray
    y_np: np.ndarray
    uncovered: set[int]
    cache: ConditionsCoverageCache
    prediction_quality_metric: Callable[
        [SurvivalRuleSet, np.ndarray, np.ndarray, np.ndarray], float
    ]
    finished: dict[str, bool]

    def clone(self) -> RuleSetGrowingContext:

        return RuleSetGrowingContext(
            params=self.params,
            X=self.X,
            y=self.y,
            X_np=self.X_np,
            y_np=self.y_np,
            cache=self.cache,
            prediction_quality_metric=self.prediction_quality_metric,
            # those cannot be copied as they are mutable
            uncovered=set(self.uncovered),
            finished={key: value for key, value in self.finished.items()},
        )

    @staticmethod
    def create_initial_context(
        X: pd.DataFrame,
        y: pd.DataFrame,
        params: AlgorithmParams,
        prediction_quality_metric: Callable[[np.ndarray, np.ndarray], float],
    ) -> RuleSetGrowingContext:
        y_np: np.ndarray = y.to_numpy()
        return RuleSetGrowingContext(
            params=params,
            X=X,
            y=y,
            X_np=X.to_numpy(),
            y_np=y_np,
            uncovered=set(range(len(y_np))),
            cache=ConditionsCoverageCache(),
            prediction_quality_metric=prediction_quality_metric,
            finished={"dnf": False, "cnf": False},
        )


@dataclass
class RuleSetGrowingResult:
    type: str
    ruleset: SurvivalRuleSet
    prediction_quality: float
    context: RuleSetGrowingContext
    rule: Optional[SurvivalRule] = None


@dataclass
class RuleSetGrowingResults:
    carry_on: bool
    dnf: Optional[RuleSetGrowingResult] = None
    cnf: Optional[RuleSetGrowingResult] = None


def _calculate_rules_coverages(
    ruleset: SurvivalRuleSet,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cache: ConditionsCoverageCache,
):
    for rule in ruleset.rules:
        covered_mask: np.ndarray = _calculate_covered_mask(
            rule.premise, X_train, cache)
        uncovered_mask: np.ndarray = np.logical_not(covered_mask)
        covered_examples_indexes = np.where(covered_mask)[0]
        uncovered_examples_indexes = np.where(uncovered_mask)[0]
        survival_time: np.ndarray = X_train[:, rule.survival_time_attr_idx]
        rule.log_rank = PrecalculatedKaplanMeierEstimator.log_rank(
            survival_time,
            y_train,
            covered_examples_indexes,
            uncovered_examples_indexes,
            skip_sorting=True,
        )
        P: int = y_train.shape[0]
        N: int = 0
        p = np.count_nonzero(covered_mask)
        n = 0
        coverage = Coverage(p, n, P, N)
        rule.coverage = coverage
        if not rule.conclusion.fixed:
            rule.conclusion.estimator.fit(
                X_train[covered_mask, rule.survival_time_attr_idx],
                y_train[covered_mask],
                skip_sorting=True,
            )


def _update_ruleset(
    ruleset: SurvivalRuleSet,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cache: ConditionsCoverageCache,
):

    if len(ruleset.rules) == 0:
        raise ValueError(
            '"update" cannot be called on empty ruleset with no rules.')

    if ruleset.column_names is None:
        ruleset.column_names = X_train.columns.tolist()
    # sort data by survival time
    survival_time_attr_index = ruleset.column_names.index(
        ruleset.survival_time_attr_name
    )
    X_train, y_train = ruleset._sanitize_dataset(X_train, y_train)
    survival_time = X_train[:, survival_time_attr_index]

    # fit Kaplan Meier estimator on whole dataset as default conclusion
    ruleset.default_conclusion = SurvivalConclusion(
        value=None, column_name=ruleset.decision_attribute
    )
    ruleset.default_conclusion.estimator = PrecalculatedKaplanMeierEstimator()
    ruleset.default_conclusion.estimator.fit(
        survival_time,
        y_train,
        skip_sorting=True,  # skip sorting (dataset is already sorted)
    )
    ruleset._stored_default_conclusion = ruleset.default_conclusion
    _calculate_rules_coverages(ruleset, X_train, y_train, cache)
    # use fast log_rank to speed up calculation
    ruleset.calculate_rules_weights(PrecalculatedKaplanMeierEstimator.log_rank)


class RuleSetGrower:

    def __init__(self, original_result: RuleSetGrowingResult):
        self.ruleset: SurvivalRuleSet = original_result.ruleset
        self.last_pred_quality: float = original_result.prediction_quality
        self.initial_ctx: RuleSetGrowingContext = original_result.context
        self.cnf_ctx: RuleSetGrowingContext = original_result.context.clone()
        self.dnf_ctx: RuleSetGrowingContext = original_result.context.clone()
        self.params: AlgorithmParams = original_result.context.params
        self.cnf_params: CNFParams = to_cnf_params(self.params)
        self.dnf_params: DNFParams = to_dnf_params(self.params)
        self.y: pd.Series = None

        self.cnf_inducer = CNFInducer(
            {
                **self.cnf_params,
                "max_conjunction_length": self.params["max_component_length"],
            }
        )
        self.dnf_inducer = DNFInducer(
            {
                **self.dnf_params,
                "max_disjunction_length": self.params["max_component_length"],
            }
        )
        survival_time_column_index = original_result.context.X.columns.tolist().index(
            self.params["survival_time_attr"]
        )
        self.cnf_inducer.survival_time_column_index = survival_time_column_index
        self.dnf_inducer.survival_time_column_index = survival_time_column_index

        # share the cache between inducers to save memory and reduce training times
        self.cache: ConditionsCoverageCache = self.initial_ctx.cache
        self.dnf_inducer.cache = self.cache
        self.cnf_inducer.cache = self.cache
        self.logger: Logger = getLogger(self.__class__.__name__)
        self.logger.disabled = True

    def grow_single_rule(self) -> RuleSetGrowingResults:
        """Adds a single rule to the ruleset. It returns two new rulesets, which are
        the original ruleset with the new DNF and CNF rule added to it.

        Returns:
            RuleSetGrowingResults: object containing two grown rulesets and their
            qualities. Those rulesets are the original ruleset with the new DNF and
            CNF rule added to it.
        """
        self.logger.info("Inducing rules")
        rules_choices: RuleChoices = self._induce_both_rules(
            (self.initial_ctx.X, self.initial_ctx.y),
            (self.initial_ctx.X_np, self.initial_ctx.y_np),
        )
        # if we have finished induction for all classes for algorithms, we can stop
        carry_on = not self.dnf_ctx.finished["dnf"] or not self.cnf_ctx.finished["cnf"]
        # Now select the best rule and add it to the ruleset
        if carry_on:
            dnf_result: RuleSetGrowingResult = self.prepare_result(
                rules_choices.dnf)
            cnf_result: RuleSetGrowingResult = self.prepare_result(
                rules_choices.cnf)
            return RuleSetGrowingResults(
                dnf=dnf_result, cnf=cnf_result, carry_on=carry_on
            )
        return RuleSetGrowingResults(
            dnf=RuleSetGrowingResult(
                type="dnf",
                rule=None,
                ruleset=None,
                prediction_quality=float("-inf"),
                context=self.dnf_ctx,
            ),
            cnf=RuleSetGrowingResult(
                type="cnf",
                rule=None,
                ruleset=None,
                prediction_quality=float("-inf"),
                context=self.cnf_ctx,
            ),
            carry_on=False,
        )

    def prepare_result(self, choice: RuleChoice) -> Optional[RuleSetGrowingResult]:
        context: RuleSetGrowingContext = (
            self.dnf_ctx if choice.type == "dnf" else self.cnf_ctx
        )
        if choice.rule is None:
            context.finished[choice.type] = True
            return RuleSetGrowingResult(
                type=choice.type,
                rule=None,
                ruleset=None,
                prediction_quality=float("-inf"),
                context=context,
            )
        rule: SurvivalRule = choice.rule
        ruleset: SurvivalRuleSet = self.clone_ruleset(self.ruleset)
        ruleset.rules.append(choice.rule)
        _update_ruleset(ruleset, self.initial_ctx.X,
                        self.initial_ctx.y, self.cache)
        result = RuleSetGrowingResult(
            type=choice.type,
            rule=rule,
            ruleset=ruleset,
            prediction_quality=choice.pred_quality,
            context=context,
        )
        uncovered: set[int] = context.uncovered
        covered: set[int] = set(
            np.where(rule.premise.covered_mask(self.initial_ctx.X_np) == 1)[0]
        )
        new_uncovered: set[int] = uncovered.difference(covered)
        if len(new_uncovered) == len(uncovered):
            context.finished[choice.type] = True
        context.uncovered = new_uncovered
        return result

    def _induce_both_rules(
        self,
        dataset: tuple[pd.DataFrame, pd.Series],
        dataset_np: tuple[np.ndarray, np.ndarray],
    ) -> RuleChoices:
        X, y = dataset
        X_np, y_np = dataset_np
        self.y = y

        dnf_rule: Optional[SurvivalRule] = None
        dnf_pred_quality: float = float("-inf")
        cnf_rule: Optional[SurvivalRule] = None
        cnf_pred_quality: float = float("-inf")

        # Induce best rules: one in DNF and one in CNF form
        column_names: list[str] = list(X.columns)
        dnf_rule = SurvivalRule(
            premise=CompoundCondition(
                subconditions=[], logic_operator=LogicOperators.CONJUNCTION
            ),
            conclusion=SurvivalConclusion(value=np.nan, column_name=y.name),
            column_names=column_names,
            survival_time_attr=self.params["survival_time_attr"],
        )
        dnf_rule.conclusion.estimator = KaplanMeierEstimator()
        cnf_rule = SurvivalRule(
            premise=CompoundCondition(
                subconditions=[], logic_operator=LogicOperators.ALTERNATIVE
            ),
            conclusion=SurvivalConclusion(value=np.nan, column_name=y.name),
            column_names=column_names,
            survival_time_attr=self.params["survival_time_attr"],
        )
        cnf_rule.conclusion.estimator = KaplanMeierEstimator()

        def induce_rule(inducer, rule, X, X_np, y_np, uncovered):
            return inducer._grow(  # pylint: disable=protected-access
                rule, X, X_np, y_np, uncovered
            )

        results_list = Parallel(n_jobs=2, prefer="processes")(
            delayed(induce_rule)(**params)
            for params in [
                {
                    "inducer": self.dnf_inducer,
                    "rule": dnf_rule,
                    "X": X,
                    "X_np": X_np,
                    "y_np": y_np,
                    "uncovered": self.cnf_ctx.uncovered,
                },
                {
                    "inducer": self.cnf_inducer,
                    "rule": cnf_rule,
                    "X": X,
                    "X_np": X_np,
                    "y_np": y_np,
                    "uncovered": self.dnf_ctx.uncovered,
                },
            ]
        )
        ((dnf_rule, dnf_carry_on), (cnf_rule, cnf_carry_on)) = results_list

        if not cnf_carry_on:
            self.logger.info("No more CNF rules can be induced")
            cnf_rule = None
            self.cnf_ctx.finished["cnf"] = True
            self.dnf_ctx.finished["cnf"] = True
        if not dnf_carry_on:
            self.logger.info("No more DNF rules can be induced")
            dnf_rule = None
            self.cnf_ctx.finished["dnf"] = True
            self.dnf_ctx.finished["dnf"] = True

        if cnf_carry_on and self.cnf_params["enable_pruning"]:
            uncovered: set[int] = self.cnf_ctx.uncovered
            self.cnf_inducer._prune(cnf_rule, uncovered, X_np, y_np)
            # if cnf_rule.voting_weight is None or np.isnan(cnf_rule.voting_weight):
            #     raise Exception("CNF rule has nan voting weight")

            cnf_pred_quality: float = (
                self.check_predictive_quality_on_ruleset_with_rule(
                    self.ruleset,
                    cnf_rule,
                    X,
                    self.cnf_ctx.prediction_quality_metric,
                )
            )
            self.logger.debug(
                'Induced CNF rule: "%s" with pred quality: ' "%f",
                str(cnf_rule),
                cnf_pred_quality,
            )

        if dnf_carry_on and self.dnf_params["enable_pruning"]:
            uncovered: set[int] = self.dnf_ctx.uncovered
            self.dnf_inducer._prune(dnf_rule, uncovered, X_np, y_np)
            # if dnf_rule.voting_weight is None or np.isnan(dnf_rule.voting_weight):
            #     raise Exception("DNF rule has nan voting weight")
            dnf_pred_quality: float = (
                self.check_predictive_quality_on_ruleset_with_rule(
                    self.ruleset,
                    dnf_rule,
                    X,
                    self.dnf_ctx.prediction_quality_metric,
                )
            )
            self.logger.debug(
                'Induced DNF rule: "%s" with pred quality: ' "%f",
                str(dnf_rule),
                dnf_pred_quality,
            )

        return RuleChoices(
            dnf=RuleChoice("dnf", dnf_rule, dnf_pred_quality),
            cnf=RuleChoice("cnf", cnf_rule, cnf_pred_quality),
        )

    def check_predictive_quality_on_ruleset_with_rule(
        self,
        ruleset: SurvivalRuleSet,
        rule: SurvivalRule,
        X: pd.DataFrame,
        prediction_quality_metric: Callable[
            [SurvivalRuleSet, np.ndarray, np.ndarray], float
        ],
    ) -> float:
        new_ruleset: SurvivalRuleSet = ruleset.__class__(
            ruleset.rules + [rule], survival_time_attr=self.params["survival_time_attr"]
        )
        # Here is better to update whole ruleset to make sure conclusions are updated
        for rule in new_ruleset.rules:
            rule.conclusion.estimator = KaplanMeierEstimator()
        _update_ruleset(new_ruleset, X, self.y, self.cache)
        # new_ruleset.default_conclusion = ruleset.default_conclusion
        return prediction_quality_metric(new_ruleset, X, self.y)

    def clone_ruleset(self, ruleset: SurvivalRuleSet) -> SurvivalRuleSet:
        clone = SurvivalRuleSet(
            rules=[r for r in ruleset.rules],
            survival_time_attr=self.params["survival_time_attr"],
        )
        clone.default_conclusion = SurvivalConclusion(
            value=ruleset.default_conclusion.value,
            column_name=ruleset.default_conclusion.column_name,
            fixed=ruleset.default_conclusion.fixed,
        )
        clone.estimator = ruleset.default_conclusion.estimator
        return clone
