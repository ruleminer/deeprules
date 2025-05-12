import json
import os
import pathlib

import numpy as np
import pandas as pd
import pytest
import utils
from decision_rules import conditions
from decision_rules import measures
from decision_rules.classification import ClassificationRuleSet
from decision_rules.serialization import JSONSerializer
from decision_rules.serialization import SerializationModes
from sklearn import metrics

from deeprules.classification.mixed import Classifier

dir_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture
def heart_c_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    return utils.read_dataset(problem_type="classification", dataset_name="heart-c")


def test_classifier(
    heart_c_dataset: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
):
    model = Classifier()
    X_train, y_train, X_test, y_test = heart_c_dataset
    ruleset: ClassificationRuleSet = model.fit(X_train, y_train)

    assert ruleset.decision_attribute is not None
    assert len(ruleset.rules) < 11
    assert metrics.balanced_accuracy_score(
        y_test, ruleset.predict(X_test)) > 0.82

    ruleset_path: str = dir_path / "json" / "heart_c_ruleset.json"
    with open(ruleset_path, "r", encoding="utf-8") as f:
        expected_ruleset: ClassificationRuleSet = JSONSerializer.deserialize(
            json.load(f), target_class=ClassificationRuleSet
        )
    assert ruleset == expected_ruleset
    expected_ruleset.calculate_condition_importances(
        X_train, y_train, measure=measures.c2)
    # with open(ruleset_path, 'w+', encoding='utf-8') as f:
    #     json.dump(JSONSerializer.serialize(ruleset, mode='full'), f, indent=2)


def test_serialization(
    heart_c_dataset: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
):
    model = Classifier(voting_measure=measures.c2)
    X_train, y_train, _, _ = heart_c_dataset
    X_train, y_train = X_train.iloc[:20], y_train.iloc[:20]
    ruleset: ClassificationRuleSet = model.fit(X_train, y_train)

    assert len(ruleset.rules) > 0
    ruleset.rules[0].premise.subconditions += [
        conditions.DiscreteSetCondition(
            column_index=2, values_set={"asympt", "non_anginal"}
        ),
        conditions.NominalAttributesEqualityCondition(
            column_indices=[10, 1, 2]),
        conditions.AttributesRelationCondition(
            column_left=1, operator=">", column_right=2
        ),
    ]
    ruleset.update(X_train, y_train, measures.c2)

    min_serialized_ruleset: dict = JSONSerializer.serialize(
        ruleset, mode=SerializationModes.MINIMAL
    )
    min_serialized_ruleset: ClassificationRuleSet = JSONSerializer.deserialize(
        min_serialized_ruleset, ClassificationRuleSet
    )
    min_serialized_ruleset.update(X_train, y_train, measures.c2)
    assert np.array_equal(
        ruleset.predict(X_train), min_serialized_ruleset.predict(X_train)
    )

    full_serialized_ruleset: dict = JSONSerializer.serialize(
        ruleset, mode=SerializationModes.FULL
    )
    full_deserialized_ruleset: ClassificationRuleSet = JSONSerializer.deserialize(
        full_serialized_ruleset, ClassificationRuleSet
    )

    assert np.array_equal(
        ruleset.predict(X_train), full_deserialized_ruleset.predict(X_train)
    )
