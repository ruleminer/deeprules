"""Contains custom conditions extending AbstractCondition class from
`decision-rules <https://github.com/ruleminer/decision-rules>`_ package. It also
contains their JSON serializer classes and some utility functions concerning conditions.

"""
from __future__ import annotations

from decision_rules.conditions import ElementaryCondition
from decision_rules.conditions import NominalCondition
from decision_rules.core.condition import AbstractCondition


def does_condition_support_negation(condition: AbstractCondition):
    """Returns whether given condition support negation

    Args:
        condition (AbstractCondition): condition to check
    Returns:
        bool: whether given condition support negation
    """
    return isinstance(condition, (ElementaryCondition, NominalCondition))
