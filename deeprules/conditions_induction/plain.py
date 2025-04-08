import numpy as np
from decision_rules.conditions import ElementaryCondition, NominalCondition
from decision_rules.core.condition import AbstractCondition

from deeprules.conditions_induction._base import AbstractConditionsGenerator


class PlainConditionsInducer(AbstractConditionsGenerator):

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        nominal_attributes_without_empty_values: dict[int, np.ndarray],
        nominal_attributes_indices: list[int],
        numerical_attributes_indices: list[int],
        enable_negations: bool,
        enable_intervals: bool,
    ) -> None:
        super().__init__(
            X=X,
            y=y,
            nominal_attributes_without_empty_values=nominal_attributes_without_empty_values,
            nominal_attributes_indices=nominal_attributes_indices,
            numerical_attributes_indices=numerical_attributes_indices,
        )
        self.enable_negations: bool = enable_negations
        self.enable_intervals: bool = enable_intervals

    def generate(
        self,
        examples_covered_by_rule: np.ndarray,
        y: np.ndarray,
        mid_points: dict[int, np.ndarray],
    ) -> list[AbstractCondition]:
        conditions: list[AbstractCondition] = []
        conditions.extend(self._generate_nominal_conditions())
        conditions.extend(self._generate_numerical_conditions(mid_points=mid_points))
        return conditions

    def _generate_nominal_conditions(self) -> list[AbstractCondition]:
        conditions: list[NominalCondition] = []

        for indx in self.nominal_attributes_indices:
            column: np.ndarray = self.nominal_attributes_without_empty_values[indx]
            conditions.extend(
                [
                    NominalCondition(column_index=indx, value=val)
                    for val in np.unique(column)
                ]
            )

        if self.enable_negations:
            negated_conditions = []
            for condition in conditions:
                negated_condition = NominalCondition(
                    column_index=condition.column_index,
                    value=condition.value,
                )
                negated_condition.negated = True
                negated_conditions.append(negated_condition)
            conditions.extend(negated_conditions)

        return conditions

    def _generate_numerical_conditions(
        self, mid_points: dict[int, np.ndarray]
    ) -> list[AbstractCondition]:
        conditions: list[ElementaryCondition] = []

        for indx in self.numerical_attributes_indices:
            attr_mid_points: np.ndarray = mid_points[indx]
            conditions.extend(
                [
                    ElementaryCondition(
                        column_index=indx,
                        left_closed=False,
                        right_closed=True,
                        left=float("-inf"),
                        right=mid_point,
                    )
                    for mid_point in attr_mid_points
                ]
            )
            conditions.extend(
                [
                    ElementaryCondition(
                        column_index=indx,
                        left_closed=True,
                        right_closed=False,
                        left=mid_point,
                        right=float("inf"),
                    )
                    for mid_point in attr_mid_points
                ]
            )

        if self.enable_intervals:
            conditions.extend(self._generate_interval_conditions(mid_points))

        return conditions

    def _generate_interval_conditions(
        self, mid_points: dict[int, np.ndarray]
    ) -> list[AbstractCondition]:
        conditions = []
        for indx in self.numerical_attributes_indices:
            keys = mid_points[indx]
            for i, left in enumerate(keys[1:-2]):
                for right in keys[i:-1]:
                    moved_left_value: float = keys[i - 1] + (
                        (keys[i] - keys[i - 1]) / 2
                    )
                    conditions.append(
                        ElementaryCondition(
                            column_index=indx,
                            left_closed=True,
                            right_closed=True,
                            left=moved_left_value,
                            right=right,
                        )
                    )

        return conditions
