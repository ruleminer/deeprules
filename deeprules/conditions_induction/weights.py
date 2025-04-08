from decision_rules.conditions import (AttributesCondition,
                                       ElementaryCondition, NominalCondition)
from decision_rules.core.condition import AbstractCondition

from deeprules.conditions import NominalAttributesEqualityCondition


def get_condition_weight(condition: AbstractCondition) -> float:
    """Get condition preferability sorting weights. The higher the weight, the more
    preferable the condition is.

    Based on:
        https://github.com/adaa-polsl/m-of-n-rules/blob/master/src/Rulekit_complex_reg/adaa.analytics.rules/src/main/java/adaa/analytics/rules/logic/induction/AbstractFinder.java#L353

    Args:
        condition (AbstractCondition): condition instance

    Returns:
        float: preferability sorting weights
    """
    if isinstance(condition, ElementaryCondition) or isinstance(
        condition, NominalCondition
    ):
        # plain condition
        return 2.0 if condition.negated else 3.0
    if isinstance(condition, AttributesCondition) or isinstance(
        condition, NominalAttributesEqualityCondition
    ):
        # attributes relation condition
        return 0.0 if condition.negated else 1.0
    raise ValueError(f"Unknown condition type: {condition.__class__.__name__}")

def get_more_preferable_condition(c1: AbstractCondition, c2: AbstractCondition) -> AbstractCondition:
    if c1 is None and c2 is not None:
        return c2
    if c2 is None and c1 is not None:
        return c1
    return max([
        (c1, get_condition_weight(c1)),
        (c2, get_condition_weight(c2))
    ], key=lambda x: x[1])[0]