from typing import TypedDict

from deeprules._params import QualityMeasure
from deeprules.classification.cnf._params import AlgorithmParams as CNFParams
from deeprules.classification.dnf._params import AlgorithmParams as DNFParams


class AlgorithmParams(TypedDict):
    min_cov: int
    max_layers_count: int
    max_component_length: int

    dnf_quality_measure: QualityMeasure
    dnf_pruning_measure: QualityMeasure
    dnf_select_best_candidate_measure: QualityMeasure

    cnf_quality_measure: QualityMeasure
    cnf_pruning_measure: QualityMeasure
    cnf_select_best_candidate_measure: QualityMeasure

    voting_measure: QualityMeasure  # voting weight is same for both cnf and dnf
    enable_pruning: bool

    enable_attributes_conditions: bool
    enable_negations: bool
    enable_discrete_set_conditions: bool


def to_cnf_params(params: AlgorithmParams) -> CNFParams:
    return CNFParams(
        min_cov=params["min_cov"],
        max_layers_count=params["max_layers_count"],
        max_conjunction_length=params["max_component_length"],
        quality_measure=params["cnf_quality_measure"],
        pruning_measure=params["cnf_pruning_measure"],
        select_best_candidate_measure=params["cnf_select_best_candidate_measure"],
        enable_pruning=params["enable_pruning"],
        enable_attributes_conditions=params["enable_attributes_conditions"],
        enable_negations=params["enable_negations"],
    )


def to_dnf_params(params: AlgorithmParams) -> DNFParams:
    return DNFParams(
        min_cov=params["min_cov"],
        max_layers_count=params["max_layers_count"],
        max_disjunction_length=params["max_component_length"],
        quality_measure=params["dnf_quality_measure"],
        pruning_measure=params["dnf_pruning_measure"],
        select_best_candidate_measure=params["dnf_select_best_candidate_measure"],
        enable_pruning=params["enable_pruning"],
        enable_attributes_conditions=params["enable_attributes_conditions"],
        enable_negations=params["enable_negations"],
    )
