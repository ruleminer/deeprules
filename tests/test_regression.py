import pandas as pd
import pytest
import utils
from decision_rules.regression import RegressionRuleSet
from sklearn import metrics

from deeprules.regression.mixed import Regressor


@pytest.fixture
def cholesterol_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    return utils.read_dataset(problem_type="regression", dataset_name="cholesterol")


def test_regressor(
    cholesterol_dataset: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
):
    model = Regressor()
    X_train, y_train, X_test, y_test = cholesterol_dataset
    ruleset: RegressionRuleSet = model.fit(X_train, y_train)
    assert len(ruleset.rules) < 17
    assert (
        metrics.mean_absolute_percentage_error(y_test, ruleset.predict(X_test)) < 0.1821
    )
