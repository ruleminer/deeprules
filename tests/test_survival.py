import pandas as pd
import pytest
import utils
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
