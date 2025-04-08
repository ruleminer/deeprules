import pandas as pd
from decision_rules.problem import ProblemTypes
from decision_rules.survival import SurvivalRuleSet

from deeprules._model import BaseModel
from deeprules._params import DEFAULT_PARAMS_VALUES
from deeprules.survival.mixed._induction import GreedyRuleInducer


class  Survival( BaseModel):
    """Survival rules based on  Rules algorithm."""

    _Inducer =  GreedyRuleInducer
    _problem_type = ProblemTypes.SURVIVAL

    def __init__(
        self,
        min_cov: int = DEFAULT_PARAMS_VALUES["min_cov"],
        max_uncovered_fraction: float = DEFAULT_PARAMS_VALUES["max_uncovered_fraction"],
        max_layers_count: int = DEFAULT_PARAMS_VALUES["max_layers_count"],
        max_component_length: int = 3,
        enable_pruning: bool = DEFAULT_PARAMS_VALUES["enable_pruning"],
        enable_attributes_conditions: bool = DEFAULT_PARAMS_VALUES[
            "enable_attributes_conditions"
        ],
        enable_negations: bool = DEFAULT_PARAMS_VALUES["enable_negations"],
        survival_time_attr: str = "survival_time",
    ):  # pylint: disable=unused-argument
        params: dict = locals()
        params.pop("self")
        super().__init__(**params)

        self.training_history: pd.DataFrame = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SurvivalRuleSet:
        super().fit(X, y)
        inducer:  GreedyRuleInducer = self._inducer
        self.training_history = pd.DataFrame(inducer.history)
        return self.ruleset
