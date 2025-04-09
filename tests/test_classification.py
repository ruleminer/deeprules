import os

import pandas as pd
import pytest
import utils
from decision_rules.classification import ClassificationRuleSet
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

    assert len(ruleset.rules) < 9
    assert (
        metrics.balanced_accuracy_score(y_test, ruleset.predict(X_test)) > 0.802
    )
