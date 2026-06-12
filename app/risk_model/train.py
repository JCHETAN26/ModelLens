"""Train a real credit-risk classifier on the German Credit dataset.

Logistic regression is chosen deliberately: with standardized numeric features
and one-hot categoricals, each feature's `coefficient x encoded value` is an
exact, signed contribution to the risk score — which maps directly onto
ModelLens factors (direction + weight + evidence). No SHAP needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.risk_model.dataset import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RISK_LABEL,
    TARGET,
    load_german_credit,
)


@dataclass
class TrainedRiskModel:
    """A fitted pipeline plus held-out evaluation metrics."""

    pipeline: Pipeline
    auc: float
    accuracy: float
    n_train: int
    n_test: int
    # Held-out rows kept around so we can generate explanation cases from
    # applicants the model did not train on.
    test_frame: pd.DataFrame

    def metrics(self) -> dict[str, float | int]:
        return {
            "roc_auc": round(self.auc, 4),
            "accuracy": round(self.accuracy, 4),
            "n_train": self.n_train,
            "n_test": self.n_test,
        }


def _build_pipeline(seed: int) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def train_risk_model(seed: int = 42, test_size: float = 0.25) -> TrainedRiskModel:
    """Train and evaluate the credit-risk model. Deterministic for a given seed."""
    frame = load_german_credit()
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x = frame[features].copy()
    # Encode the risk class as the positive label (1 = bad / higher risk).
    y = (frame[TARGET] == RISK_LABEL).astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )

    pipeline = _build_pipeline(seed)
    pipeline.fit(x_train, y_train)

    proba = pipeline.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    auc = float(roc_auc_score(y_test, proba))
    acc = float(accuracy_score(y_test, preds))

    test_frame = x_test.copy()
    test_frame[TARGET] = frame.loc[x_test.index, TARGET].values

    return TrainedRiskModel(
        pipeline=pipeline,
        auc=auc,
        accuracy=acc,
        n_train=len(x_train),
        n_test=len(x_test),
        test_frame=test_frame.reset_index(drop=True),
    )


if __name__ == "__main__":  # pragma: no cover
    model = train_risk_model()
    print("Trained credit-risk model (German Credit):")
    for k, v in model.metrics().items():
        print(f"  {k}: {v}")
