from typing import TypedDict


class AlgorithmParams(TypedDict):
    min_cov: int
    max_uncovered_fraction: float
    max_disjunction_length: int
    max_layers_count: int
    enable_pruning: bool
    survival_time_attr: str = "survival_time"

    enable_attributes_conditions: bool
    enable_negations: bool
