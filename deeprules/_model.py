from typing import Any
from typing import Optional
from typing import Type

import pandas as pd
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.problem import ProblemTypes
from sklearn.base import BaseEstimator

from deeprules._induction import RuleInducersMixin
from deeprules._induction import RuleInductionTimes
from deeprules._params import adjust_params_on_dataset


class BaseModel(BaseEstimator):

    _Inducer: Type[RuleInducersMixin] = None
    _problem_type: ProblemTypes = None

    def __init__(self, **algorithm_params: dict):
        if self._Inducer is None:
            raise NotImplementedError(
                "_Inducer field must point to valid class implementing "
                "RuleInducersMixin."
            )
        if self._problem_type is None:
            raise NotImplementedError(
                "_problem_type field must point to value from "
                "decision_rules.problem.ProblemTypes enum"
            )

        self._params: dict[str, Any] = algorithm_params
        self.induction_times: RuleInductionTimes = None
        self.ruleset: Optional[AbstractRuleSet] = None
        self._inducer: RuleInducersMixin = None

    def set_params(self, **params):
        self._params.update(params)

    def get_params(self, deep=True) -> dict:
        return self._params

    def fit(self, X: pd.DataFrame, y: pd.Series) -> AbstractRuleSet:
        """Trains a ruleset on given data.

        Args:
            X (pd.DataFrame): dataset
            y (pd.Series): label column

        Returns:
            AbstractRuleSet: trained ruleset instance from `decision_rules <https://github.com/ruleminer/decision-rules>`_ package.
        """
        adjusted_params: dict[str, Any] = adjust_params_on_dataset(
            self._params, y, problem_type=self._problem_type
        )
        self._inducer: RuleInducersMixin = self._Inducer(
            adjusted_params
        )  # pylint: disable=not-callable
        self.ruleset: AbstractRuleSet = self._inducer.induce_ruleset(X, y)
        self.induction_times = self._inducer.induction_times
        self.ruleset.decision_attribute = y.name
        return self.ruleset
