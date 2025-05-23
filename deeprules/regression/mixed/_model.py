import pandas as pd
from decision_rules import measures
from decision_rules.problem import ProblemTypes
from decision_rules.regression import RegressionRuleSet

from deeprules._model import BaseModel
from deeprules._params import DEFAULT_PARAMS_VALUES
from deeprules._params import QualityMeasure
from deeprules.regression.mixed._induction import GreedyRuleInducer


class Regressor(BaseModel):
    """Regressor based on deeprules algorithm."""

    _Inducer = GreedyRuleInducer
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
        enable_discrete_set_conditions: bool = DEFAULT_PARAMS_VALUES[
            "enable_discrete_set_conditions"
        ],
    ):
        """
        Args:
            min_cov (int, optional): A minimum number of previously uncovered
                examples to be covered by a new rule (positive examples for
                classification problems); Defaults to DEFAULT_PARAMS_VALUES["min_cov"].
            max_uncovered_fraction (float, optional): Floating-point number from [0,1]
                interval representing maximum fraction of examples that may remain
                uncovered by the rule set. Defaults to
                DEFAULT_PARAMS_VALUES["max_uncovered_fraction"].
            max_layers_count (int, optional): Maximum number of top level components
                (conjunctions or disjunctions) in rule . Defaults to
                DEFAULT_PARAMS_VALUES["max_layers_count"].
            max_component_length (int, optional): Maximum number of conditions in each
                top level component (conjunction or disjunction). Defaults to 3.
            dnf_quality_measure (QualityMeasure, optional): Quality measure used for DNF
                rules induction. Defaults to measures.correlation.
            dnf_pruning_measure (QualityMeasure, optional): Quality measure used for DNF
                rules pruning. Defaults to DEFAULT_PARAMS_VALUES["pruning_measure"].
            dnf_select_best_candidate_measure (QualityMeasure, optional): Quality
                measure used for selecting best condition when growing DNF rules.
                Defaults to measures.c2.
            cnf_quality_measure (QualityMeasure, optional): Quality measure used for DNF
                rules induction. Defaults to measures.correlation. Defaults to
                measures.c2.
            cnf_pruning_measure (QualityMeasure, optional): Quality measure used for CNF
                rules pruning. Defaults to DEFAULT_PARAMS_VALUES["pruning_measure"].
            cnf_select_best_candidate_measure (QualityMeasure, optional): Quality
                measure used for selecting best condition when growing DNF rules.
                Defaults to measures.correlation.
            voting_measure (QualityMeasure, optional): Quality measure used for solving
                voting conflicts. Defaults to DEFAULT_PARAMS_VALUES["voting_measure"].
            enable_pruning (bool, optional): Enables pruning. Defaults to
                DEFAULT_PARAMS_VALUES["enable_pruning"].
            enable_attributes_conditions (bool, optional): Enables attributes relations
                conditions. Such conditions take the following form "attr1 > y" or
                "attr1 = attr2". Defaults to
                DEFAULT_PARAMS_VALUES[ "enable_attributes_conditions" ].
            enable_negations (bool, optional): Enables negated conditions in rules.
                Defaults to DEFAULT_PARAMS_VALUES["enable_negations"].
            enable_discrete_set_conditions (bool, optional): Enables discrete set
                conditions. Such conditions take the following form "x in {1,2,3}". They
                are generated only for nominal attributes. Defaults to
                DEFAULT_PARAMS_VALUES["enable_discrete_set_conditions"].
        """
        # pylint: disable=unused-argument
        params: dict = locals()
        params.pop("self")
        super().__init__(**params)

        self.training_history: pd.DataFrame = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RegressionRuleSet:
        super().fit(X, y)
        inducer: GreedyRuleInducer = self._inducer
        self.training_history = pd.DataFrame(inducer.history)
        return self.ruleset
