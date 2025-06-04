import json
import os
import pathlib

import numpy as np
import pandas as pd
import pytest
import utils
from decision_rules import conditions
from decision_rules import measures
from decision_rules.regression import RegressionRuleSet
from decision_rules.serialization import JSONSerializer
from decision_rules.serialization import SerializationModes
from sklearn import metrics

from deeprules.regression.mixed import Regressor

dir_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture
def cholesterol_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    return utils.read_dataset(problem_type="regression", dataset_name="cholesterol")


def test_regressor(
    cholesterol_dataset: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
):
    model = Regressor()
    X_train, y_train, X_test, y_test = cholesterol_dataset
    ruleset: RegressionRuleSet = model.fit(X_train, y_train)
    assert ruleset.decision_attribute is not None
    assert len(ruleset.rules) < 17
    mape = metrics.mean_absolute_percentage_error(
        y_test, ruleset.predict(X_test))
    assert (
        mape < 0.183
    )
    ruleset_path: str = dir_path / "json" / "reg_ruleset.json"
    with open(ruleset_path, "r", encoding="utf-8") as f:
        expected_ruleset: RegressionRuleSet = JSONSerializer.deserialize(
            json.load(f), target_class=RegressionRuleSet
        )
    assert ruleset == expected_ruleset
    expected_ruleset.calculate_condition_importances(
        X_train, y_train, measure=measures.c2
    )
    # with open(ruleset_path, 'w+', encoding='utf-8') as f:
    #     json.dump(JSONSerializer.serialize(ruleset, mode='full'), f, indent=2)


def test_serialization(
    cholesterol_dataset: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
):
    model = Regressor()
    X_train, y_train, _, _ = cholesterol_dataset
    X_train, y_train = X_train.iloc[:20], y_train.iloc[:20]
    ruleset: RegressionRuleSet = model.fit(X_train, y_train)

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
    min_deserialized_ruleset: RegressionRuleSet = JSONSerializer.deserialize(
        min_serialized_ruleset, RegressionRuleSet
    )
    assert ruleset == min_deserialized_ruleset

    min_deserialized_ruleset.update(X_train, y_train, measures.c2)
    assert np.array_equal(
        ruleset.predict(X_train), min_deserialized_ruleset.predict(X_train)
    )

    full_serialized_ruleset: dict = JSONSerializer.serialize(
        ruleset, mode=SerializationModes.FULL
    )
    full_deserialized_ruleset: RegressionRuleSet = JSONSerializer.deserialize(
        full_serialized_ruleset, RegressionRuleSet
    )
    assert ruleset == full_deserialized_ruleset

    assert np.array_equal(
        ruleset.predict(X_train), full_deserialized_ruleset.predict(X_train)
    )
