from typing import Callable, TypeAlias, TypedDict
import pandas as pd
from decision_rules import measures
from decision_rules.core.coverage import Coverage
from decision_rules.problem import ProblemTypes
import math

QualityMeasure: TypeAlias = Callable[[Coverage], float]


class BaseAlgorithmParams(TypedDict):
    min_cov: int
    max_uncovered_fraction: float
    max_layers_count: int
    quality_measure: QualityMeasure
    pruning_measure: QualityMeasure
    voting_measure: QualityMeasure
    select_best_candidate_measure: QualityMeasure
    enable_pruning: bool

    enable_attributes_conditions: bool
    enable_negations: bool


DEFAULT_PARAMS_VALUES: BaseAlgorithmParams = BaseAlgorithmParams(
    min_cov=5,
    max_uncovered_fraction=0.0,
    max_layers_count=10,
    quality_measure=measures.c2,
    pruning_measure=measures.c2,
    voting_measure=measures.c2,
    select_best_candidate_measure=measures.c2,
    enable_pruning=True,
    enable_attributes_conditions=True,
    enable_negations=True,
)


def adjust_params_on_dataset(
    params: BaseAlgorithmParams,
    y: pd.Series,
    problem_type: ProblemTypes,
) -> BaseAlgorithmParams:
    if problem_type != ProblemTypes.CLASSIFICATION:
        return params.copy()
    new_params: BaseAlgorithmParams = params.copy()
    minority_class_size: int = y.value_counts().min()
    new_params["min_cov"] = math.ceil(min(minority_class_size, params["min_cov"]))
    return new_params
