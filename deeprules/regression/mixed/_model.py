import pandas as pd
from decision_rules import measures
from decision_rules.problem import ProblemTypes
from decision_rules.regression import RegressionRuleSet

from deeprules._model import BaseModel
from deeprules._params import DEFAULT_PARAMS_VALUES, QualityMeasure
from deeprules.regression.mixed._induction import GreedyRuleInducer


class  Regressor( BaseModel):
    """Regressor based on  Rules algorithm."""

    _Inducer =  GreedyRuleInducer
    _problem_type = ProblemTypes.REGRESSION

    def __init__(
        self,
        min_cov: int = DEFAULT_PARAMS_VALUES["min_cov"],
        max_uncovered_fraction: float = DEFAULT_PARAMS_VALUES["max_uncovered_fraction"],
        max_layers_count: int = DEFAULT_PARAMS_VALUES["max_layers_count"],
        max_component_length: int = 3,
        dnf_quality_measure: QualityMeasure = measures.correlation,
        dnf_pruning_measure: QualityMeasure = DEFAULT_PARAMS_VALUES["pruning_measure"],
        dnf_select_best_candidate_measure: QualityMeasure = measures.c2,
        cnf_quality_measure: QualityMeasure = measures.c2,
        cnf_pruning_measure: QualityMeasure = DEFAULT_PARAMS_VALUES["pruning_measure"],
        cnf_select_best_candidate_measure: QualityMeasure = measures.correlation,
        voting_measure: QualityMeasure = DEFAULT_PARAMS_VALUES["voting_measure"],
        enable_pruning: bool = DEFAULT_PARAMS_VALUES["enable_pruning"],
        enable_attributes_conditions: bool = DEFAULT_PARAMS_VALUES[
            "enable_attributes_conditions"
        ],
        enable_negations: bool = DEFAULT_PARAMS_VALUES["enable_negations"],
    ):  # pylint: disable=unused-argument
        params: dict = locals()
        params.pop("self")
        super().__init__(**params)

        self.training_history: pd.DataFrame = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RegressionRuleSet:
        super().fit(X, y)
        inducer:  GreedyRuleInducer = self._inducer
        self.training_history = pd.DataFrame(inducer.history)
        return self.ruleset
