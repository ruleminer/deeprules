from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from typing import Callable

import pandas as pd
from decision_rules.core.ruleset import AbstractRuleSet

from deeprules import _helpers
from deeprules._timing import PerformanceTimer
from deeprules.conditions_induction import ConditionsGenerator


@dataclass
class RuleInductionTimes:
    growing_time: timedelta = timedelta()
    pruning_time: timedelta = timedelta()
    total_training_time: timedelta = timedelta()

    def __add__(self, other: RuleInductionTimes) -> RuleInductionTimes:
        if other == 0:
            return self
        if not isinstance(other, RuleInductionTimes):
            raise TypeError(f"Cannot add {type(other)} to RuleInductionTimes")
        return RuleInductionTimes(
            growing_time=self.growing_time + other.growing_time,
            pruning_time=self.pruning_time + other.pruning_time,
            total_training_time=self.total_training_time + other.total_training_time,
        )

    def __radd__(self, other: RuleInductionTimes) -> RuleInductionTimes:
        return self.__add__(other)

    def __repr__(self) -> str:
        return (
            f"growing_time={self.growing_time.total_seconds()}, "
            f"pruning_time={self.pruning_time.total_seconds()}, "
            f"total_training_time={self.total_training_time.total_seconds()}"
        )


class RuleInducersMixin(ABC):

    def __init__(self):
        self.params: dict[str, Any] = None
        self.induction_times: RuleInductionTimes = RuleInductionTimes()
        self.condition_generator: ConditionsGenerator = None
        self._setup_timers()

    @abstractmethod
    def induce_ruleset(self, X: pd.DataFrame, y: pd.Series) -> AbstractRuleSet:
        raise NotImplementedError(
            "RuleInducersMixin requires induce_ruleset method to be implemented"
        )

    @abstractmethod
    def _grow(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def _prune(self, *args, **kwargs) -> Any:
        pass

    def _setup_timers(self):
        self._setup_timer_for_method(
            "induce_ruleset", save_to="total_training_time")
        self._setup_timer_for_method("_grow", save_to="growing_time")
        self._setup_timer_for_method("_prune", save_to="pruning_time")

    def _setup_timer_for_method(self, method_name: str, save_to: str):
        method: Callable = getattr(self, method_name, None)
        if method is None:
            raise ValueError(
                f"RuleInducesMixin requires {method_name} method to be implemented"
            )

        def wrapped_method(*args, **kwargs):
            with PerformanceTimer() as timer:
                result: Any = method(*args, **kwargs)
            # store time
            new_timedelta: timedelta = (
                getattr(self.induction_times, save_to) + timer.timedelta
            )
            setattr(self.induction_times, save_to, new_timedelta)
            return result

        setattr(self, method_name, wrapped_method)

    def _setup_condition_generator(self, X: pd.DataFrame, y: pd.Series):
        self.condition_generator = ConditionsGenerator(
            X,
            y,
            numerical_attributes_indices=_helpers.get_numerical_indexes(X),
            nominal_attributes_indices=_helpers.get_nominal_indexes(X),
            cuts_only_between_classes=True,
            enable_negations=self.params["enable_negations"],
            enable_attributes_conditions=self.params["enable_attributes_conditions"],
        )
