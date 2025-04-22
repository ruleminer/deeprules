import numpy as np
import pandas as pd
import pytest
import utils
from decision_rules import conditions
from decision_rules import measures
from decision_rules.serialization import JSONSerializer
from decision_rules.serialization import SerializationModes
from decision_rules.survival import SurvivalRuleSet

from deeprules.survival.mixed import Survival


@pytest.fixture
def mgus_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    return utils.read_dataset(problem_type="survival", dataset_name="mgus")


def test_survival(
    mgus_dataset: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
):
    model = Survival(survival_time_attr="survival_time")
    X_train, y_train, X_test, y_test = mgus_dataset
    ruleset: SurvivalRuleSet = model.fit(X_train, y_train)
    assert ruleset.decision_attribute is not None
    ibs: float = ruleset.integrated_bier_score(
        X_test, y_test, ruleset.predict(X_test))
    assert len(ruleset.rules) < 5
    assert ibs < 0.1768


def test_serialization(
    mgus_dataset: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
):
    model = Survival(survival_time_attr="survival_time")
    X_train, y_train, _, _ = mgus_dataset
    X_train, y_train = X_train.iloc[:20], y_train.iloc[:20]
    ruleset: SurvivalRuleSet = model.fit(X_train, y_train)

    assert len(ruleset.rules) > 0
    ruleset.rules[0].premise.subconditions += [
        conditions.DiscreteSetCondition(column_index=2, values_set={"3", "4"}),
        conditions.NominalAttributesEqualityCondition(
            column_indices=[1, 2, 4]),
        conditions.AttributesRelationCondition(
            column_left=0, operator="<", column_right=6
        ),
    ]

    min_serialized_ruleset: dict = JSONSerializer.serialize(
        ruleset, mode=SerializationModes.MINIMAL
    )
    min_deserialized_ruleset: SurvivalRuleSet = JSONSerializer.deserialize(
        min_serialized_ruleset, SurvivalRuleSet
    )
    assert ruleset == min_deserialized_ruleset

    # TODO: This will fail due to known bug in decision-rules. Fix it and uncomment
    # those lines.

    # min_deserialized_ruleset.update(X_train, y_train)
    # assert np.array_equal(
    #     ruleset.predict(X_train), min_deserialized_ruleset.predict(X_train)
    # )

    full_serialized_ruleset: dict = JSONSerializer.serialize(
        ruleset, mode=SerializationModes.FULL
    )
    full_deserialized_ruleset: SurvivalRuleSet = JSONSerializer.deserialize(
        full_serialized_ruleset, SurvivalRuleSet
    )
    assert ruleset == full_deserialized_ruleset

    assert all([
        (
            np.array_equal(a['times'], b['times']) and
            np.array_equal(a['median_survival_time'],
                           b['median_survival_time'])
        ) for a, b in zip(
            ruleset.predict(
                X_train), full_deserialized_ruleset.predict(X_train)
        )
    ])
