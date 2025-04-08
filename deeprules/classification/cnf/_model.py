import pandas as pd
from decision_rules.classification import ClassificationRuleSet
from decision_rules.problem import ProblemTypes

from deeprules._model import BaseModel
from deeprules._params import DEFAULT_PARAMS_VALUES, QualityMeasure
from deeprules.classification.cnf._induction import RuleInducer


class  ClassifierCNF( BaseModel):
    """Classifier based on  Rules algorithm. It produces CNF rules in the following
    form:
        IF (a1 AND a2 ... AND aN) OR (b1 AND b2 ... AND bN) THEN label = ...

    Where maximum number of literals in each conjunction could be controlled
    by :code:`max_conjunction_length` parameter. Maximum number of conjunctions
    in a rule could be controlled by :code:`max_layers_count` parameter.
    """

    _Inducer =  RuleInducer
    _problem_type = ProblemTypes.CLASSIFICATION

    def __init__(
        self,
        min_cov: int = DEFAULT_PARAMS_VALUES["min_cov"],
        max_uncovered_fraction: float = DEFAULT_PARAMS_VALUES['max_uncovered_fraction'],
        max_layers_count: int = DEFAULT_PARAMS_VALUES["max_layers_count"],
        max_conjunction_length: int = 3,
        quality_measure: QualityMeasure = DEFAULT_PARAMS_VALUES["quality_measure"],
        pruning_measure: QualityMeasure = DEFAULT_PARAMS_VALUES["pruning_measure"],
        voting_measure: QualityMeasure = DEFAULT_PARAMS_VALUES["voting_measure"],
        select_best_candidate_measure: QualityMeasure = DEFAULT_PARAMS_VALUES[
            "select_best_candidate_measure"
        ],
        enable_pruning: bool = DEFAULT_PARAMS_VALUES["enable_pruning"],
        enable_attributes_conditions: bool = DEFAULT_PARAMS_VALUES[
            "enable_attributes_conditions"
        ],
        enable_negations: bool = DEFAULT_PARAMS_VALUES["enable_negations"],
    ):  # pylint: disable=unused-argument
        params: dict = locals()
        params.pop("self")
        super().__init__(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ClassificationRuleSet:
        return super().fit(X, y)
