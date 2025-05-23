from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from logging import Logger
from typing import Callable
from typing import Optional

import numpy as np
import pandas as pd
from decision_rules.classification import ClassificationConclusion
from decision_rules.classification import ClassificationRule
from decision_rules.classification import ClassificationRuleSet
from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import LogicOperators
from joblib import delayed
from joblib import Parallel

from deeprules.cache import ConditionsCoverageCache
from deeprules.classification.cnf._induction import RuleInducer as CNFInducer
from deeprules.classification.cnf._params import AlgorithmParams as CNFParams
from deeprules.classification.dnf._induction import RuleInducer as DNFInducer
from deeprules.classification.dnf._params import AlgorithmParams as DNFParams
from deeprules.classification.mixed._params import AlgorithmParams
from deeprules.classification.mixed._params import to_cnf_params
from deeprules.classification.mixed._params import to_dnf_params


@dataclass
class RuleChoice:
    type: str
    rule: Optional[ClassificationRule]
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
    class_uncovered: dict[str, set[int]]
    finished_classes: dict[str, set[str]]
    cache: ConditionsCoverageCache
    prediction_quality_metric: Callable[[np.ndarray, np.ndarray], float]
    classes: np.ndarray
    classes_sizes: np.ndarray

    def clone(self) -> RuleSetGrowingContext:

        return RuleSetGrowingContext(
            params=self.params,
            X=self.X,
            y=self.y,
            X_np=self.X_np,
            y_np=self.y_np,
            cache=self.cache,
            prediction_quality_metric=self.prediction_quality_metric,
            classes=self.classes,
            classes_sizes=self.classes_sizes,
            # those cannot be copied as they are mutable
            class_uncovered={
                class_value: set(uncovered)
                for class_value, uncovered in self.class_uncovered.items()
            },
            finished_classes={
                "dnf": set(self.finished_classes["dnf"]),
                "cnf": set(self.finished_classes["cnf"]),
            },
        )

    @staticmethod
    def create_initial_context(
        X: pd.DataFrame,
        y: pd.DataFrame,
        params: AlgorithmParams,
        prediction_quality_metric: Callable[[np.ndarray, np.ndarray], float],
    ) -> RuleSetGrowingContext:
        classes, classes_sizes = np.unique(y, return_counts=True)
        # Start with the minority class to achieve a working single rule classifier
        # the default rule is always a majority class so that way we can predict
        sorted_indices: np.ndarray = np.argsort(classes_sizes)
        classes, classes_sizes = (
            classes[sorted_indices],
            classes_sizes[sorted_indices],
        )
        y_np: np.ndarray = y.to_numpy()
        class_uncovered: dict[str, set[int]] = {
            class_value: set(np.where(y_np == class_value)[0])
            for class_value in classes
        }
        return RuleSetGrowingContext(
            params=params,
            X=X,
            y=y,
            X_np=X.to_numpy(),
            y_np=y.to_numpy(),
            class_uncovered=class_uncovered,
            finished_classes={"dnf": set(), "cnf": set()},
            cache=ConditionsCoverageCache(),
            prediction_quality_metric=prediction_quality_metric,
            classes=classes,
            classes_sizes=classes_sizes,
        )


@dataclass
class RuleSetGrowingResult:
    type: str
    ruleset: ClassificationRuleSet
    prediction_quality: float
    context: RuleSetGrowingContext
    rule: Optional[ClassificationRule] = None


@dataclass
class RuleSetGrowingResults:
    carry_on: bool
    dnf: Optional[RuleSetGrowingResult] = None
    cnf: Optional[RuleSetGrowingResult] = None


class RuleSetGrower:

    def __init__(self, original_result: RuleSetGrowingResult):
        self.ruleset: ClassificationRuleSet = original_result.ruleset
        self.last_pred_quality: float = original_result.prediction_quality
        self.initial_ctx: RuleSetGrowingContext = original_result.context
        self.cnf_ctx: RuleSetGrowingContext = original_result.context.clone()
        self.dnf_ctx: RuleSetGrowingContext = original_result.context.clone()
        self.params: AlgorithmParams = original_result.context.params
        self.cnf_params: CNFParams = to_cnf_params(self.params)
        self.dnf_params: DNFParams = to_dnf_params(self.params)

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
        self.cnf_inducer._setup_condition_generator(
            self.initial_ctx.X, self.initial_ctx.y
        )
        self.dnf_inducer._setup_condition_generator(
            self.initial_ctx.X, self.initial_ctx.y
        )

        # share the cache between inducers to save memory and reduce training times
        self.dnf_inducer.cache = self.initial_ctx.cache
        self.logger: Logger = getLogger(self.__class__.__name__)
        self.logger.disabled = True

    def grow_single_rule(
        self, class_value: str, class_size: int
    ) -> RuleSetGrowingResults:
        """Adds a single rule to the ruleset. It returns two new rulesets, which are
        the original ruleset with the new DNF and CNF rule added to it.

        Returns:
            RuleSetGrowingResults: object containing two grown rulesets and their
            qualities. Those rulesets are the original ruleset with the new DNF and
            CNF rule added to it.
        """
        self.logger.info("Inducing rules for class %s", class_value)
        rules_choices: RuleChoices = self._induce_both_rules(
            class_value,
            class_size,
            (self.initial_ctx.X, self.initial_ctx.y),
            (self.initial_ctx.X_np, self.initial_ctx.y_np),
        )
        # if we have finished induction for all classes for algorithms, we can stop
        carry_on = len(self.dnf_ctx.finished_classes["dnf"]) != len(
            self.dnf_ctx.classes_sizes
        ) or len(self.cnf_ctx.finished_classes["cnf"]) != len(
            self.cnf_ctx.classes_sizes
        )
        # Now select the best rule and add it to the ruleset
        if carry_on:
            dnf_result: RuleSetGrowingResult = self.prepare_result(
                rules_choices.dnf, class_value
            )
            cnf_result: RuleSetGrowingResult = self.prepare_result(
                rules_choices.cnf, class_value
            )
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

    def prepare_result(
        self,
        choice: RuleChoice,
        class_value: str,
    ) -> Optional[RuleSetGrowingResult]:
        context: RuleSetGrowingContext = (
            self.dnf_ctx if choice.type == "dnf" else self.cnf_ctx
        )
        if choice.rule is None:
            context.finished_classes[choice.type].add(class_value)
            return RuleSetGrowingResult(
                type=choice.type,
                rule=None,
                ruleset=None,
                prediction_quality=float("-inf"),
                context=context,
            )
        rule: ClassificationRule = choice.rule
        ruleset: ClassificationRuleSet = self.clone_ruleset(self.ruleset)
        ruleset.rules.append(choice.rule)
        ruleset.update(
            self.initial_ctx.X, self.initial_ctx.y, self.params["voting_measure"]
        )
        result = RuleSetGrowingResult(
            type=choice.type,
            rule=rule,
            ruleset=ruleset,
            prediction_quality=choice.pred_quality,
            context=context,
        )
        uncovered: set[int] = context.class_uncovered[class_value]
        covered: set[int] = set(
            np.where(rule.premise.covered_mask(self.initial_ctx.X_np) == 1)[0]
        )
        new_uncovered: set[int] = uncovered.difference(covered)
        if len(new_uncovered) == len(uncovered):
            context.finished_classes[choice.type].add(class_value)
        context.class_uncovered[class_value] = new_uncovered
        return result

    def _induce_both_rules(
        self,
        class_value: str,
        class_size: int,
        dataset: tuple[pd.DataFrame, pd.Series],
        dataset_np: tuple[np.ndarray, np.ndarray],
    ) -> RuleChoices:
        X, y = dataset
        X_np, y_np = dataset_np

        dnf_rule: Optional[ClassificationRule] = None
        dnf_pred_quality: float = float("-inf")
        cnf_rule: Optional[ClassificationRule] = None
        cnf_pred_quality: float = float("-inf")

        # Induce best rules: one in DNF and one in CNF form
        dnf_rule = ClassificationRule(
            premise=CompoundCondition(
                subconditions=[], logic_operator=LogicOperators.CONJUNCTION
            ),
            conclusion=ClassificationConclusion(
                value=class_value, column_name=y.name),
            column_names=list(X.columns),
        )
        cnf_rule = ClassificationRule(
            premise=CompoundCondition(
                subconditions=[], logic_operator=LogicOperators.ALTERNATIVE
            ),
            conclusion=ClassificationConclusion(
                value=class_value, column_name=y.name),
            column_names=list(X.columns),
        )

        def induce_rule(inducer, rule, X, X_np, y_np, uncovered, class_size):
            return inducer._grow(  # pylint: disable=protected-access
                rule,
                X,
                X_np,
                y_np,
                uncovered,
                class_size,
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
                    "uncovered": self.dnf_ctx.class_uncovered[class_value],
                    "class_size": class_size,
                },
                {
                    "inducer": self.cnf_inducer,
                    "rule": cnf_rule,
                    "X": X,
                    "X_np": X_np,
                    "y_np": y_np,
                    "uncovered": self.cnf_ctx.class_uncovered[class_value],
                    "class_size": class_size,
                },
            ]
        )
        ((dnf_rule, dnf_carry_on), (cnf_rule, cnf_carry_on)) = results_list

        if not cnf_carry_on:
            self.logger.info(
                "No more CNF rules can be induced for class %s", class_value
            )
            cnf_rule = None
            self.cnf_ctx.finished_classes["cnf"].add(class_value)
            self.dnf_ctx.finished_classes["cnf"].add(class_value)
        if not dnf_carry_on:
            self.logger.info(
                "No more DNF rules can be induced for class %s", class_value
            )
            dnf_rule = None
            self.cnf_ctx.finished_classes["dnf"].add(class_value)
            self.dnf_ctx.finished_classes["dnf"].add(class_value)

        if cnf_carry_on and self.cnf_params["enable_pruning"]:
            uncovered: set[int] = self.cnf_ctx.class_uncovered[class_value]
            self.cnf_inducer._prune(
                cnf_rule, uncovered, class_size, X_np, y_np)
            cnf_rule.voting_weight = self.params["voting_measure"](
                cnf_rule.coverage)
            cnf_pred_quality: float = (
                self.check_predictive_quality_on_ruleset_with_rule(
                    self.ruleset,
                    cnf_rule,
                    X,
                    y_np,
                    self.cnf_ctx.prediction_quality_metric,
                )
            )
            self.logger.debug(
                'Induced CNF rule: "%s" with pred quality: ' "%f",
                str(cnf_rule),
                cnf_pred_quality,
            )

        if dnf_carry_on and self.dnf_params["enable_pruning"]:
            uncovered: set[int] = self.dnf_ctx.class_uncovered[class_value]
            self.dnf_inducer._prune(
                dnf_rule, uncovered, class_size, X_np, y_np)
            dnf_rule.voting_weight = self.params["voting_measure"](
                dnf_rule.coverage)
            dnf_pred_quality: float = (
                self.check_predictive_quality_on_ruleset_with_rule(
                    self.ruleset,
                    dnf_rule,
                    X,
                    y_np,
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
        ruleset: ClassificationRuleSet,
        rule: ClassificationRule,
        X: pd.DataFrame,
        y: np.ndarray,
        prediction_quality_metric: Callable[[np.ndarray, np.ndarray], float],
    ) -> float:
        new_ruleset: ClassificationRuleSet = ruleset.__class__(
            ruleset.rules + [rule])
        # TODO: A bit overkill, no need to update whole ruleset
        new_ruleset.update(X, y, measure=self.params["voting_measure"])
        new_ruleset.default_conclusion = ruleset.default_conclusion
        return prediction_quality_metric(y, new_ruleset.predict(X))

    def clone_ruleset(self, ruleset: ClassificationRuleSet) -> ClassificationRuleSet:
        clone = ClassificationRuleSet(rules=[r for r in ruleset.rules])
        clone.default_conclusion = ClassificationConclusion(
            value=ruleset.default_conclusion.value,
            column_name=ruleset.default_conclusion.column_name,
        )
        return clone
