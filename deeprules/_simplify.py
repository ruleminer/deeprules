from collections import defaultdict
from copy import deepcopy

import numpy as np
from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import DiscreteSetCondition
from decision_rules.conditions import ElementaryCondition
from decision_rules.conditions import LogicOperators
from decision_rules.conditions import NominalCondition
from decision_rules.core.condition import AbstractCondition
from decision_rules.core.ruleset import AbstractRuleSet


def simplify_ruleset(
    ruleset: AbstractRuleSet, search_for_discrete_set_conditions: bool = True
):
    ruleset = deepcopy(ruleset)
    for rule in ruleset.rules:
        rule.premise = _simplify_condition(
            rule.premise, search_for_discrete_set_conditions
        )
    return ruleset


def _merge_intervals(condition: CompoundCondition):
    if (
        not isinstance(condition, CompoundCondition)
        or condition.logic_operator != LogicOperators.CONJUNCTION
    ):
        return
    conditions_map: dict[int, list[ElementaryCondition]
                         ] = defaultdict(lambda: set())
    columns_intervals: dict[
        int, tuple[list[bool], list[float], list[float], list[bool]]
    ] = defaultdict(lambda: ([], [], [], []))
    for subcondition in condition.subconditions:
        if isinstance(subcondition, ElementaryCondition) and (
            subcondition.left == float(
                "-inf") or subcondition.right == float("inf")
        ):
            conditions_map[subcondition.column_index].add(subcondition)
    for column_index, conditions in conditions_map.items():
        if len(conditions) < 2:
            continue
        left_closed, lefts, rights, right_closed = columns_intervals[column_index]
        for c in conditions:
            if c.negated:
                if c.left is not None and c.left != float("-inf"):
                    left_closed.append(False)
                    lefts.append(float("-inf"))
                    rights.append(c.left)
                    right_closed.append(not c.left_closed)
                if c.right is not None and c.right != float("inf"):
                    left_closed.append(not c.right_closed)
                    lefts.append(c.right)
                    rights.append(float("inf"))
                    right_closed.append(False)
            else:
                left_closed.append(c.left_closed)
                lefts.append(c.left if c.left is not None else float("-inf"))
                rights.append(c.right if c.right is not None else float("inf"))
                right_closed.append(c.right_closed)
        # find max left and min right
        left_closed = np.array(left_closed)
        right_closed = np.array(right_closed)
        lefts = np.array(lefts)
        rights = np.array(rights)
        max_left: float = max(lefts)
        min_right: float = min(rights)
        is_left_closed = all(left_closed[np.where(lefts == max_left)])
        is_right_closed = all(right_closed[np.where(rights == min_right)])

        # remove reduced conditions
        index_to_put: int = -1
        for i, c in enumerate(conditions):
            if i == 0:
                index_to_put = condition.subconditions.index(c)
            condition.subconditions.remove(c)

        new_condition = ElementaryCondition(
            column_index=column_index,
            left=max_left,
            right=min_right,
            left_closed=is_left_closed,
            right_closed=is_right_closed,
        )
        condition.subconditions.insert(index_to_put, new_condition)


def _simplify_condition(
    condition: AbstractCondition, search_for_discrete_set_conditions: bool
) -> AbstractCondition:
    if not isinstance(condition, CompoundCondition):
        return condition
    if search_for_discrete_set_conditions:
        _make_discrete_set_conditions(condition)
    # for single element conjunction
    if len(condition.subconditions) == 1:
        is_negated = condition.negated
        condition = condition.subconditions[0]
        if is_negated:
            condition.negated = not condition.negated
        return condition
    else:
        condition.subconditions = [
            _simplify_condition(c, search_for_discrete_set_conditions)
            for c in condition.subconditions
        ]
    # remove duplicates
    tmp = []
    for c in condition.subconditions:
        if c not in tmp:
            tmp.append(c)
    condition.subconditions = tmp
    # Merge intervals
    _merge_intervals(condition)
    # I and II De Morgan law
    if condition.negated and all(
        not isinstance(s, CompoundCondition) for s in condition.subconditions
    ):
        for s in condition.subconditions:
            s.negated = not s.negated
        condition.negated = False
        condition.logic_operator = {
            LogicOperators.ALTERNATIVE: LogicOperators.CONJUNCTION,
            LogicOperators.CONJUNCTION: LogicOperators.ALTERNATIVE,
        }[condition.logic_operator]

    tmp = []
    index_to_put = -1
    for s in condition.subconditions:
        if (
            isinstance(s, CompoundCondition)
            and s.logic_operator == condition.logic_operator
            and not s.negated
        ):
            tmp.append(s)
    for e in tmp:
        index_to_put = condition.subconditions.index(e)
        condition.subconditions = (
            condition.subconditions[:index_to_put]
            + e.subconditions
            + condition.subconditions[index_to_put + 1:]
        )

    return condition


def _make_discrete_set_conditions(condition: AbstractCondition):
    if not isinstance(condition, CompoundCondition):
        return condition
    if condition.logic_operator != LogicOperators.ALTERNATIVE:
        return
    if len(condition.subconditions) == 0 or isinstance(
        condition.subconditions[0], CompoundCondition
    ):
        return
    attr_conditions: dict[str, list[NominalCondition]] = defaultdict(list)
    attr_conditions_negated: dict[str,
                                  list[NominalCondition]] = defaultdict(list)
    remaining_conditions: list[AbstractCondition] = []
    new_conditions = []
    for c in condition.subconditions:
        if isinstance(c, NominalCondition):
            if c.negated:
                attr_conditions_negated[c.column_index].append(c)
            else:
                attr_conditions[c.column_index].append(c)
        else:
            remaining_conditions.append(c)

    for negated, conditions_dict in [
        (True, attr_conditions_negated),
        (False, attr_conditions),
    ]:
        for key, items in conditions_dict.items():
            if len(items) == 1:
                remaining_conditions += items
            else:
                cond = DiscreteSetCondition(
                    column_index=key, values_set=set(c.value for c in items)
                )
                cond.negated = negated
                new_conditions.append(cond)
    condition.subconditions = new_conditions + remaining_conditions
