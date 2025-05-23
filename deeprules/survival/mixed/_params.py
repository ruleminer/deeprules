from typing import TypedDict

from deeprules.survival.cnf._params import AlgorithmParams as CNFParams
from deeprules.survival.dnf._params import AlgorithmParams as DNFParams


class AlgorithmParams(TypedDict):
    min_cov: int
    max_layers_count: int
    max_component_length: int
    enable_pruning: bool
    enable_attributes_conditions: bool
    enable_negations: bool
    enable_discrete_set_conditions: bool

    survival_time_attr: str = "survival_time"


def to_cnf_params(params: AlgorithmParams) -> CNFParams:
    return CNFParams(
        min_cov=params["min_cov"],
        max_layers_count=params["max_layers_count"],
        max_conjunction_length=params["max_component_length"],
        enable_pruning=params["enable_pruning"],
        enable_attributes_conditions=params["enable_attributes_conditions"],
        enable_negations=params["enable_negations"],
        survival_time_attr=params["survival_time_attr"],
    )


def to_dnf_params(params: AlgorithmParams) -> DNFParams:
    return DNFParams(
        min_cov=params["min_cov"],
        max_layers_count=params["max_layers_count"],
        max_disjunction_length=params["max_component_length"],
        enable_pruning=params["enable_pruning"],
        enable_attributes_conditions=params["enable_attributes_conditions"],
        enable_negations=params["enable_negations"],
        survival_time_attr=params["survival_time_attr"],
    )
