from __future__ import annotations

import numpy as np
from decision_rules.conditions import ElementaryCondition, NominalCondition
from decision_rules.core.condition import AbstractCondition
from decision_rules.serialization._core.conditions import (
    _AttributesConditionSerializer, _BaseConditionModel,
    _BaseConditionSerializer, _CompoundConditionSerializer,
    _ConditionSerializer, _ElementaryConditionSerializer,
    _NominalConditionSerializer)
from decision_rules.serialization.utils import (JSONClassSerializer,
                                                register_serializer)


def does_condition_support_negation(condition: AbstractCondition):
    """Returns whether given condition support negation

    Args:
        condition (AbstractCondition): condition to check
    Returns:
        bool: whether given condition support negation
    """
    return isinstance(condition, (ElementaryCondition, NominalCondition))

class NominalAttributesEqualityCondition(AbstractCondition):
    """Class that represents condition where nominal attributes 
    are compared for equality

    Example:
        IF **attr1 = attr2 = attr3** THEN y = 1
    """

    def __init__(
        self,
        column_indices: list[str],
    ):
        super().__init__()
        self.column_indices: list[str] = column_indices

    @property
    def attributes(self) -> frozenset[int]:
        return frozenset(self.column_indices)

    def _calculate_covered_mask(self, X: np.ndarray) -> np.ndarray:
        arrays = X[:, list(self.column_indices)].T
        # mask = (
        #     (arrays[:-1] == arrays[1:])
        # )[0]
        # return mask & np.logical_not(np.any(arrays == None, axis=0))
        tmp = X[:, list(self.column_indices)[0]]
        mask = None
        for i in list(self.column_indices[1:]):
            if mask is None:
                mask = tmp == X[:, i]
                continue
            submask = tmp == X[:, i]
            mask = mask & submask
        return mask & np.logical_not(np.any(arrays == None, axis=0))

    def to_string(self, columns_names: list[str]) -> str:
        operator: str = ' != ' if self.negated else ' = '
        return operator.join(
            columns_names[index] for index in self.column_indices
        )

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, self.__class__):
            return False
        return (
            __o.column_indices == self.column_indices and
            __o.negated == self.negated
        )

    def __hash__(self):
        return hash((
            self.negated,
            self.attributes,
        ))


class DiscreteSetCondition(AbstractCondition):
    """Class that represents condition where discrete attributes value 
    belongs to a given set of values.

    Example:
        IF **color = {red, green}** THEN y = 1
    """

    def __init__(
        self,
        column_index: str,
        values_set: set[str],
    ):
        super().__init__()

        self.column_index: str = column_index
        self.values_set: set[str] = values_set

    @property
    def attributes(self) -> frozenset[int]:
        return frozenset({self.column_index})

    def _calculate_covered_mask(self, X: np.ndarray) -> np.ndarray:
        return np.any([
            X[:, self.column_index] == e for e in self.values_set
        ], axis=0)

    def to_string(self, columns_names: list[str]) -> str:
        column_name: str = columns_names[self.column_index]
        return column_name + ' = {' + ', '.join(self.values_set) + '}'

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, self.__class__):
            return False
        return (
            __o.column_index == self.column_index and
            __o.values_set == self.values_set and
            __o.negated == self.negated
        )

    def __hash__(self):
        return hash((
            self.negated,
            self.attributes,
            self.values_set
        ))


class _DiscreteSetConditionSerializer(_BaseConditionSerializer, JSONClassSerializer):

    condition_type: str = 'discrete_set'
    condition_class: type = DiscreteSetCondition

    class _Model(_BaseConditionModel):
        values_set: set[str]

    @staticmethod
    def _from_pydantic_model(model: _DiscreteSetConditionSerializer._Model) -> DiscreteSetCondition:
        condition = DiscreteSetCondition(
            column_index=model.attributes[0],
            values_set=model.values_set,
        )
        condition.negated = model.negated if model.negated is not None else False
        return condition

    @staticmethod
    def _to_pydantic_model(instance: DiscreteSetCondition) -> _DiscreteSetConditionSerializer._Model:
        return _DiscreteSetConditionSerializer._Model(
            type=_DiscreteSetConditionSerializer.condition_type,
            attributes=list(instance.attributes),
            negated=instance.negated,
            values_set=instance.values_set
        )


class _NominalAttributesEqualityConditionSerializer(_BaseConditionSerializer, JSONClassSerializer):

    condition_type: str = 'nominal_attributes_equality'
    condition_class: type = NominalAttributesEqualityCondition

    class _Model(_BaseConditionModel):
        pass

    @staticmethod
    def _from_pydantic_model(model: _NominalAttributesEqualityConditionSerializer._Model) -> DiscreteSetCondition:
        condition = NominalAttributesEqualityCondition(
            column_indices=set(model.attributes),
        )
        condition.negated = model.negated if model.negated is not None else False
        return condition

    @staticmethod
    def _to_pydantic_model(instance: DiscreteSetCondition, **kwargs) -> _NominalAttributesEqualityConditionSerializer._Model:
        return _NominalAttributesEqualityConditionSerializer._Model(
            type=_NominalAttributesEqualityConditionSerializer.condition_type,
            attributes=list(instance.attributes),
            negated=instance.negated,
        )


_ConditionSerializer._elementary_conditions_serializers = [
    _NominalConditionSerializer,
    _ElementaryConditionSerializer,
    _AttributesConditionSerializer,
    _CompoundConditionSerializer,
    _DiscreteSetConditionSerializer,
    _NominalAttributesEqualityConditionSerializer,
]

_ConditionSerializer._conditions_types_map = {
    s.condition_type: s
    for s in _ConditionSerializer._elementary_conditions_serializers
}
_ConditionSerializer._conditions_serializers_map = {
    s.condition_class: s
    for s in _ConditionSerializer._elementary_conditions_serializers
}

register_serializer(DiscreteSetCondition)(_ConditionSerializer)
register_serializer(NominalAttributesEqualityCondition)(_ConditionSerializer)
