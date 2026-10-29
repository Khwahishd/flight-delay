"""Model factories: a prior baseline, logistic regression, and gradient boosting."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL

SEED = 42


def make_prior_baseline() -> DummyClassifier:
    """Always predicts the training base rate; the floor every model must beat."""
    return DummyClassifier(strategy="prior")


def make_logistic(numeric_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            (
                "num",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric_cols,
            ),
        ]
    )
    clf = LogisticRegression(max_iter=1000, C=1.0)
    return Pipeline([("pre", pre), ("clf", clf)])


def make_hgb() -> HistGradientBoostingClassifier:
    """Handles missing weather natively and uses pandas category dtypes."""
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        learning_rate=0.1,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=SEED,
    )
