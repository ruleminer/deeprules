# DeepRules

Rule induction algorithm capable of learning rulesets with mixed CNF (and or or's) and DNF (or of and's) rules.

These packages make extensive use of the [decision-rules](https://github.com/ruleminer/decision-rules) library. All trained models are compatible with **decision-rules**.
More advanced use cases can be found in its [documentation](https://ruleminer.github.io/decision-rules/).

### Example classification ruleset:
```txt
r1: IF cp != {asympt} AND sex = {female} AND (thalach >= 109.50 OR fbs = {t}) THEN class = <50
r2: IF (cp = {asympt} OR oldpeak >= 2.50) AND ca >= 0.50 AND (trestbps >= 109.00 OR thalach >= 148.50) THEN class = >50_1
r3: IF cp != {asympt} AND (oldpeak <= 2.65 OR slope = {down}) AND (thalach >= 109.50 OR fbs = {t}) THEN class = <50
r4: IF cp = {asympt} AND (oldpeak >= 0.60 OR ca >= 0.50) AND (trestbps >= 109.00 OR thalach >= 148.50) THEN class = >50_1
r5: IF thal = {normal} AND oldpeak <= 2.65 AND (thalach >= 111.50 OR fbs = {t}) AND age <= 76.50 THEN class = <50
r6: IF (oldpeak >= 1.70 AND exang = {yes}) OR (trestbps = thalach) THEN class = >50_1
r7: IF (exang = {no} OR cp != {asympt}) AND (oldpeak <= 2.50 OR trestbps >= 167.50) AND (thalach >= 114.50 OR slope = {up}) THEN class = <50 
r8: IF (exang = {yes} AND thalach <= 156.00) OR (thalach <= 110.00 AND slope = {flat}) THEN class = >50_1
r9: IF (thal != {normal} OR oldpeak >= 2.50) AND oldpeak >= 0.50 AND (cp != {typ_angina} OR trestbps <= 120.00) THEN class = >50_1
r10: IF (slope != {up} AND age >= 46.50) OR oldpeak >= 2.40 OR (ca >= 3.00 AND restecg = {left_vent_hyper}) THEN class = >50_1
r11: IF (thalach <= 158.00 AND chol >= 242.50) OR (oldpeak >= 2.00 AND chol <= 243.00) THEN class = >50_1
```

## Example usage


For classification:
```python
import pandas as pd
from deeprules.classification import Classifier
from decision_rules.classification import ClassificationRuleSet

df: pd.DataFrame = read_dataset()
X, y = df.drop('label', axis=1), df['label']

ruleset: ClassificationRuleSet = Classifier().fit(X, y)

ruleset.predict(X)
```

For regression:
```python
import pandas as pd
from deeprules.regression.mixed import Regressor
from decision_rules.regression import RegressionRuleSet

df: pd.DataFrame = read_dataset()
X, y = df.drop('label', axis=1), df['label']

ruleset: RegressionRuleSet = Regressor().fit(X, y)

ruleset.predict(X)
```

For survival:
```python
import pandas as pd
from deeprules.survival.mixed import Survival
from decision_rules.survival import SurvivalRuleSet

df: pd.DataFrame = read_dataset()
X, y = df.drop('survival_status', axis=1), df['survival_status']
# survival status columns requires string values of '1' and '0'
y = y.astype(str).astype(int)

ruleset: SurvivalRuleSet = Survival(survival_time_attr="survival_time").fit(X, y)

ruleset.predict(X)
```

## Documentation

Documentation is placed in `docs/build/html` directory.

## Local development

Installing dev dependencies:
```bash
pip install -e .[dev]
```

Installing test dependencies:
```bash
pip install -e .[test]
```

Installing pre-commit hook:
```bash
pre-commit install
```
> First commit after installing  pre-commit may sometimes fail, just try again.

##  Running tests

```bash
pytest ./tests/
```

Running specific test:
```bash
pytest ./tests/test_classification.py::test_classifier
```
