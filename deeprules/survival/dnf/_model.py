import pandas as pd
from decision_rules.problem import ProblemTypes
from decision_rules.survival import SurvivalRuleSet

from deeprules._model import BaseModel
from deeprules._params import DEFAULT_PARAMS_VALUES
from deeprules.survival.dnf._induction import RuleInducer


class  SurvivalDNF( BaseModel):
    """Survival rules based on  Rules algorithm. It produces DNF rules in the following
    form:
        IF (a1 OR a2 ... OR aN) AND (b1 OR b2 ... OR bN) THEN label = ...

    Where maximum number of literals in each disjunction could be controlled
    by :code:`max_disjunction_length` parameter. Maximum number of disjunctions
    in a rule could be controlled by :code:`max_layers_count` parameter.
    """

    _Inducer =  RuleInducer
    _problem_type = ProblemTypes.SURVIVAL

    def __init__(
        self,
        min_cov: int = DEFAULT_PARAMS_VALUES["min_cov"],
        max_uncovered_fraction: float = DEFAULT_PARAMS_VALUES["max_uncovered_fraction"],
        max_layers_count: int = DEFAULT_PARAMS_VALUES["max_layers_count"],
        max_disjunction_length: int = 3,
        survival_time_attr: str = "survival_time",
        enable_pruning: bool = DEFAULT_PARAMS_VALUES["enable_pruning"],
        enable_attributes_conditions: bool = DEFAULT_PARAMS_VALUES[
            "enable_attributes_conditions"
        ],
        enable_negations: bool = DEFAULT_PARAMS_VALUES["enable_negations"],
    ):  # pylint: disable=unused-argument
        params: dict = locals()
        params.pop("self")
        super().__init__(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SurvivalRuleSet:
        return super().fit(X, y)
