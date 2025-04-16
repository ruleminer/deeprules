import os

import numpy as np
import pandas as pd
import pytest
import utils
from decision_rules import conditions
from decision_rules.classification import ClassificationRuleSet
from decision_rules.serialization import JSONSerializer
from decision_rules.serialization import SerializationModes
from sklearn import metrics

from deeprules.classification.mixed import Classifier

dir_path = os.path.dirname(os.path.realpath(__file__))


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
    assert len(ruleset.rules) < 9
    assert metrics.balanced_accuracy_score(
        y_test, ruleset.predict(X_test)) > 0.802


def test_serialization(
    heart_c_dataset: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
):
    model = Classifier()
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
            column_left=6, operator=">", column_right=11
        ),
    ]

    min_serialized_ruleset: dict = JSONSerializer.serialize(
        ruleset, mode=SerializationModes.MINIMAL
    )
    min_deserialized_ruleser: ClassificationRuleSet = JSONSerializer.deserialize(
        min_serialized_ruleset, ClassificationRuleSet
    )
    assert ruleset == min_deserialized_ruleser

    min_deserialized_ruleser.update(X_train, y_train)
    assert np.array_equal(
        ruleset.predict(X_train), min_deserialized_ruleser.predict(X_train)
    )

    full_serialized_ruleset: dict = JSONSerializer.serialize(
        ruleset, mode=SerializationModes.FULL
    )
    full_deserialized_ruleset: ClassificationRuleSet = JSONSerializer.deserialize(
        full_serialized_ruleset, ClassificationRuleSet
    )
    assert ruleset == full_deserialized_ruleset

    assert np.array_equal(
        ruleset.predict(X_train), full_deserialized_ruleset.predict(X_train)
    )
