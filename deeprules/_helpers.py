from typing import Optional

import numpy as np
import pandas as pd

from decision_rules.conditions import CompoundCondition
from decision_rules.core.condition import AbstractCondition


def get_bottom_level_conditions(
    condition: AbstractCondition, parent_condition: Optional[CompoundCondition] = None
) -> list[tuple[AbstractCondition, AbstractCondition]]:
    """Decompose nested compound conditions to the list of tuples where first element
    is a bottom level condition and second one is its parent condition.

    Args:
        condition (AbstractCondition): compound condition
        parent_condition (Optional[CompoundCondition], optional): Optional parent
        condition. Defaults to None.

    Returns:
        list[tuple[AbstractCondition, AbstractCondition]]: list of tuples where first
        element is a bottom level condition and second one is its parent condition.
    """
    if isinstance(condition, CompoundCondition):
        result: list[tuple[AbstractCondition, AbstractCondition]] = []
        for subcondition in condition.subconditions:
            result += get_bottom_level_conditions(subcondition, condition)
        return result
    else:
        return [(condition, parent_condition)]


def get_nominal_indexes(df: pd.DataFrame) -> list[int]:
    """Return indices of nominal columns in given dataframe

    Args:
        df (pd.DataFrame): DataFrame

    Returns:
        list[int]: list of indices of nominal columns
    """
    dtype_mask = df.dtypes == "object"
    nominal_indexes = np.where(dtype_mask)[0]
    return nominal_indexes.tolist()


def get_numerical_indexes(df: pd.DataFrame) -> list[int]:
    """Return indices of numerical columns in given dataframe

    Args:
        df (pd.DataFrame): DataFrame

    Returns:
        list[int]: list of indices of numerical columns
    """
    dtype_mask = np.logical_not(df.dtypes == "object")
    numerical_indexes = np.where(dtype_mask)[0]
    return numerical_indexes.tolist()
