from abc import ABC, abstractmethod

import numpy as np
from decision_rules.core.condition import AbstractCondition


class AbstractConditionsGenerator(ABC):

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        nominal_attributes_without_empty_values: dict[int, np.ndarray],
        nominal_attributes_indices: list[int],
        numerical_attributes_indices: list[int],
    ) -> None:
        super().__init__()
        self.X: np.ndarray = X
        self.y: np.ndarray = y
        self.nominal_attributes_without_empty_values: dict[int, np.ndarray] = (
            nominal_attributes_without_empty_values
        )
        self.nominal_attributes_indices: list[int] = nominal_attributes_indices
        self.numerical_attributes_indices: list[int] = numerical_attributes_indices

    @abstractmethod
    def generate(
        self,
        examples_covered_by_rule: np.ndarray,
        y: np.ndarray,
        mid_points: dict[int, np.ndarray],
    ) -> list[AbstractCondition]:
        pass
