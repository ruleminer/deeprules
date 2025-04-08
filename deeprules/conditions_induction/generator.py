from typing import Optional

import numpy as np
import pandas as pd
from decision_rules.core.condition import AbstractCondition

from deeprules.conditions_induction._base import AbstractConditionsGenerator
from deeprules.conditions_induction.attributes_relations import \
    AttributesRelationsConditionsGenerator
from deeprules.conditions_induction.plain import PlainConditionsInducer


class ConditionsGenerator:

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        numerical_attributes_indices: list[int],
        nominal_attributes_indices: list[int],
        cuts_only_between_classes: bool = True,
        enable_attributes_conditions: bool = False,
        attributes_conditions_forbidden_columns: Optional[list[int]] = None,
        forbidden_columns: Optional[list[int]] = None,
        enable_negations: bool = False,
        enable_intervals: bool = False,
        return_negated: bool = False
    ) -> None:
        self.return_negated: bool = return_negated
        self.X: np.ndarray = np.copy(X)
        self.y: np.ndarray = X
        self.cuts_only_between_classes: bool = cuts_only_between_classes
        self.enable_attributes_conditions: bool = enable_attributes_conditions
        self.numerical_attributes_indexes: list[int] = numerical_attributes_indices
        self.nominal_attributes_indices: list[int] = nominal_attributes_indices
        self.attributes_conditions_forbidden_columns: list[str] = attributes_conditions_forbidden_columns

        if forbidden_columns is None:
            forbidden_columns = []
        for c_index in forbidden_columns:
            if c_index in nominal_attributes_indices:
                nominal_attributes_indices.remove(c_index)
            if c_index in numerical_attributes_indices:
                numerical_attributes_indices.remove(c_index)

        self.nominal_attributes_without_empty_values: dict[int, np.ndarray] = {}
        for indx in self.nominal_attributes_indices:
            # Remove None values
            column = self.X[:, indx]
            self.nominal_attributes_without_empty_values[indx] = column[
                ~pd.isnull(column)
            ]

        self.generators: list[AbstractConditionsGenerator] = [
            PlainConditionsInducer(
                X, y,
                nominal_attributes_without_empty_values=self.nominal_attributes_without_empty_values,
                nominal_attributes_indices=nominal_attributes_indices,
                numerical_attributes_indices=numerical_attributes_indices,
                enable_negations=enable_negations,
                enable_intervals=enable_intervals
            )
        ]
        if enable_attributes_conditions:
            self.generators.append(
                AttributesRelationsConditionsGenerator(
                    X, y,
                    nominal_attributes_without_empty_values=self.nominal_attributes_without_empty_values,
                    nominal_attributes_indices=nominal_attributes_indices,
                    numerical_attributes_indices=numerical_attributes_indices,
                    enable_negations=enable_negations,
                    attributes_conditions_forbidden_columns=self.attributes_conditions_forbidden_columns,
                    max_length_of_conditions=3
                )
            )

    def _generate_mid_points(self, examples_covered_by_rule: np.ndarray, y: np.ndarray) -> dict[int, np.ndarray]:
        attributes_mid_points: dict[int, np.ndarray] = dict()
        for indx in self.numerical_attributes_indexes:
            if self.cuts_only_between_classes:
                attr_values = examples_covered_by_rule[:, indx].astype(float)
                attr_values = np.stack((attr_values, y), axis=1)
                attr_values = attr_values[~pd.isnull(attr_values[:, 0])]
                sorted_indices = np.argsort(attr_values[:, 0])
                sorted_attr_values = attr_values[sorted_indices]
                change_indices = [i for i in range(1, len(
                    sorted_attr_values)) if sorted_attr_values[i, 1] != sorted_attr_values[i-1, 1]]
                mid_points = np.unique(
                    [(sorted_attr_values[indx-1, 0] + sorted_attr_values[indx, 0]) / 2 for indx in change_indices])
            else:
                examples_covered_by_rule_for_attr = examples_covered_by_rule[:, indx].astype(
                    float)
                values = np.sort(np.unique(
                    examples_covered_by_rule_for_attr[~np.isnan(examples_covered_by_rule_for_attr)]))
                mid_points = [(x + y) / 2 for x, y in zip(values, values[1:])]
            attributes_mid_points[indx] = mid_points
        return attributes_mid_points

    def generate_conditions(
        self,
        examples_covered_by_rule: np.ndarray,
        y: np.ndarray,
    ) -> list[AbstractCondition]:
        mid_points: dict[int, np.ndarray] = self._generate_mid_points(
            examples_covered_by_rule=examples_covered_by_rule,
            y=y
        )
        conditions = []
        for generator in self.generators:
            conditions.extend(generator.generate(
                examples_covered_by_rule=examples_covered_by_rule,
                y=y,
                mid_points=mid_points
            ))
        if self.return_negated:
            for condition in conditions:
                condition.negated = not condition.negated
        return conditions
