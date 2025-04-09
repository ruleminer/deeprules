import pandas as pd
from decision_rules.classification import ClassificationRuleSet
from decision_rules.problem import ProblemTypes

from deeprules._model import BaseModel
from deeprules._params import DEFAULT_PARAMS_VALUES
from deeprules._params import QualityMeasure
from deeprules.classification.dnf._induction import RuleInducer


class ClassifierDNF(BaseModel):
    """Classifier based on  Rules algorithm. It produces DNF rules in the following
    form:
        IF (a1 OR a2 ... OR aN) AND (b1 OR b2 ... OR bN) THEN label = ...

    Where maximum number of literals in each disjunction could be controlled
    by :code:`max_disjunction_length` parameter. Maximum number of disjunctions
    in a rule could be controlled by :code:`max_layers_count` parameter.
    """

    _InducerType = RuleInducer
    _problem_type = ProblemTypes.CLASSIFICATION

    def __init__(
        self,
        min_cov: int = DEFAULT_PARAMS_VALUES["min_cov"],
        max_uncovered_fraction: float = DEFAULT_PARAMS_VALUES["max_uncovered_fraction"],
        max_layers_count: int = DEFAULT_PARAMS_VALUES["max_layers_count"],
        max_disjunction_length: int = 3,
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
            max_disjunction_length (int, optional): Maximum number of conditions in each
               disjunction. Defaults to 3.
            quality_measure (QualityMeasure, optional): Quality measure used for rules
                growing. Defaults to DEFAULT_PARAMS_VALUES["quality_measure"].
            pruning_measure (QualityMeasure, optional): Quality measure used for
                rules pruning induction. Defaults to
                DEFAULT_PARAMS_VALUES["pruning_measure"].
            voting_measure (QualityMeasure, optional): Quality measure used for solving
                voting conflicts. Defaults to DEFAULT_PARAMS_VALUES["voting_measure"].
            select_best_candidate_measure (QualityMeasure, optional): Quality measure
                used for selecting best condition when growing rules. Defaults to
                DEFAULT_PARAMS_VALUES["select_best_candidate_measure"]
            enable_pruning (bool, optional): Enables pruning. Defaults to
                DEFAULT_PARAMS_VALUES["enable_pruning"].
            enable_attributes_conditions (bool, optional): Enables attributes relations
                conditions. Such conditions take the following form "attr1 > y" or
                "attr1 = attr2". Defaults to
                DEFAULT_PARAMS_VALUES[ "enable_attributes_conditions" ].
            enable_negations (bool, optional): Enables negated conditions in rules.
                Defaults to DEFAULT_PARAMS_VALUES["enable_negations"].
        """
        # pylint: disable=unused-argument
        params: dict = locals()
        params.pop("self")
        super().__init__(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ClassificationRuleSet:
        return super().fit(X, y)
