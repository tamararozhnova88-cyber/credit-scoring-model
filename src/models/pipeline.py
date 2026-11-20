"""Leakage-safe sklearn pipeline: features fitted inside CV folds."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.features.build_features import add_features
from src.data.validation import PAY, BILL, PAID

NUMERIC = [
    "LIMIT_BAL",
    "AGE",
    *PAY,
    *BILL,
    *PAID,
    "avg_delay",
    "max_delay",
    "late_months",
    "avg_bill",
    "total_paid",
    "payment_to_bill",
]
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE", "age_group"]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_features(X)


def create_pipeline(estimator=None):
    if estimator is None:
        estimator = LogisticRegression(max_iter=1500, random_state=42)
    preprocessing = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                NUMERIC,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )
    return Pipeline(
        [("features", FeatureBuilder()), ("preprocessor", preprocessing), ("classifier", estimator)]
    )


def candidates():
    return [
        ("logistic_c1", LogisticRegression(C=1.0, max_iter=1500, random_state=42)),
        ("logistic_c01", LogisticRegression(C=0.1, max_iter=1500, random_state=42)),
        (
            "logistic_balanced",
            LogisticRegression(C=1, class_weight="balanced", max_iter=1500, random_state=42),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=120, min_samples_leaf=5, n_jobs=-1, random_state=42
            ),
        ),
        ("gradient_boosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]
