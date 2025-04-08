import itertools
from collections import defaultdict
from typing import Optional

import numpy as np
from decision_rules.conditions import AttributesCondition
from decision_rules.core.condition import AbstractCondition

from deeprules.conditions import NominalAttributesEqualityCondition
from deeprules.conditions_induction._base import AbstractConditionsGenerator


class AttributesRelationsConditionsGenerator(AbstractConditionsGenerator):

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        nominal_attributes_without_empty_values: dict[int, np.ndarray],
        nominal_attributes_indices: list[int],
        numerical_attributes_indices: list[int],
        enable_negations: bool = False,
        max_length_of_conditions: int = 3,
        attributes_conditions_forbidden_columns: Optional[list[int]] = None,
    ) -> None:
        super().__init__(
            X=X,
            y=y,
            nominal_attributes_without_empty_values=nominal_attributes_without_empty_values,
            nominal_attributes_indices=nominal_attributes_indices,
            numerical_attributes_indices=numerical_attributes_indices,
        )

        self.attributes_conditions_forbidden_columns: list[int] = (
            set()
            if attributes_conditions_forbidden_columns is None
            else set(attributes_conditions_forbidden_columns)
        )
        self.enable_negations: bool = enable_negations
        self.max_length_of_conditions: int = max_length_of_conditions
        self.possible_conditions: list[AbstractCondition] = []
        self.possible_conditions.extend(self._generate_for_nominal_attributes())
        self.possible_conditions.extend(self._generate_for_numerical_attributes())

    def generate(
        self,
        examples_covered_by_rule: np.ndarray,
        y: np.ndarray,
        mid_points: dict[int, np.ndarray],
    ) -> list[AbstractCondition]:
        # those conditions candidates never change
        return self.possible_conditions

    def _generate_for_numerical_attributes(self) -> list[AbstractCondition]:
        conditions: list[AbstractCondition] = []

        columns_to_try: list[str] = [
            e
            for e in self.numerical_attributes_indices
            if e not in self.attributes_conditions_forbidden_columns
        ]

        for column_left, column_right in itertools.combinations(columns_to_try, 2):
            conditions.extend(
                [
                    AttributesCondition(
                        column_left=column_left,
                        column_right=column_right,
                        operator=">",
                    ),
                    AttributesCondition(
                        column_left=column_left,
                        column_right=column_right,
                        operator="<",
                    ),
                    AttributesCondition(
                        column_left=column_left,
                        column_right=column_right,
                        operator="=",
                    ),
                ]
            )

        return conditions

    def _generate_for_nominal_attributes(self) -> list[AbstractCondition]:
        conditions: list[AbstractCondition] = []

        domain_gruped_attributes: dict[frozenset[str], list[int]] = defaultdict(lambda: [])
        for attr_index in self.nominal_attributes_indices:
            uniques = np.unique(
                self.nominal_attributes_without_empty_values[attr_index]
            )
            domain_gruped_attributes[frozenset(uniques)].append(attr_index)

        columns_to_try: list[int] = [
            e
            for e in self.nominal_attributes_indices
            if e not in self.attributes_conditions_forbidden_columns
        ]
        for columns_to_try in domain_gruped_attributes.values():
            combinations: list[tuple[int]] = []
            for combinations_length in range(2, self.max_length_of_conditions):
                combinations += itertools.combinations(
                    columns_to_try, r=combinations_length
                )

            for columns_indices in combinations:
                cond = NominalAttributesEqualityCondition(
                    column_indices=list(columns_indices),
                )
                conditions.append(cond)

                negated_cond = NominalAttributesEqualityCondition(
                    column_indices=list(columns_indices),
                )
                negated_cond.negated = True
                conditions.append(negated_cond)
        return conditions

    # def _generate_for_nominal_attributes(self) -> list[AbstractCondition]:
    #     conditions: list[AbstractCondition] = []

    #     attr_uniques: dict[int, np.ndarray] = {
    #         attr_index: np.unique(
    #             self.nominal_attributes_without_empty_values[attr_index]
    #         )
    #         for attr_index in self.nominal_attributes_indices
    #     }
    #     # get only binary attributes
    #     binary_attr_indices: set[int] = {
    #         attr_index
    #         for attr_index, uniques in attr_uniques.items()
    #         if len(uniques) == 2
    #     }

    #     columns_to_try: list[int] = [
    #         e for e in self.nominal_attributes_indices
    #         if e not in self.attributes_conditions_forbidden_columns
    #     ]
    #     combinations: list[tuple[int]] = []
    #     for combinations_length in range(2, self.max_length_of_conditions):
    #         combinations += itertools.combinations(
    #             columns_to_try, r=combinations_length
    #         )

    #     for columns_indices in combinations:
    #         # check if all attributes have the same possible values
    #         uniques = [attr_uniques[i] for i in columns_indices]
    #         if not all(np.array_equal(uniques[0], e) for e in uniques):
    #             continue

    #         cond = NominalAttributesEqualityCondition(
    #             column_indices=list(columns_indices),
    #         )
    #         conditions.append(cond)

    #         # negations are only allowed for binary attributes
    #         if self.enable_negations and all(i in binary_attr_indices for i in columns_indices):
    #             negated_cond = NominalAttributesEqualityCondition(
    #                 column_indices=list(columns_indices),
    #             )
    #             negated_cond.negated = True
    #             conditions.append(negated_cond)
    #     return conditions
