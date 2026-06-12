"""Convert a trained risk model's prediction on one applicant into a RiskCase.

This is the bridge between the real ML model and the ModelLens explanation
pipeline: the model's per-feature contributions become the structured `factors`
(direction + weight + evidence) that the LangGraph workflow then explains and
the evaluators check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.risk_model.dataset import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    readable,
)
from app.risk_model.train import TrainedRiskModel
from app.schemas.risk_case import FactorDirection, RiskBand, RiskCase, RiskFactor

# Drop factors whose absolute contribution is negligible.
_MIN_ABS_CONTRIBUTION = 1e-6


def _band(score: float) -> RiskBand:
    if score >= 0.66:
        return RiskBand.HIGH
    if score >= 0.33:
        return RiskBand.MEDIUM
    return RiskBand.LOW


def _evidence(feature: str, value: object, z: float | None) -> str:
    name = readable(feature)
    if feature in NUMERIC_FEATURES and z is not None:
        position = "above" if z > 0 else "below"
        return (
            f"{name} is {value}, which is {position} the typical applicant "
            f"for this portfolio."
        )
    return f"{name} is '{value}' for this applicant."


def _contributions(
    model: TrainedRiskModel, row: pd.Series
) -> list[tuple[str, object, float, float | None]]:
    """Return (feature, raw_value, signed_contribution, z) per original feature."""
    pre = model.pipeline.named_steps["pre"]
    clf = model.pipeline.named_steps["clf"]
    coef = clf.coef_[0]

    row_df = row.to_frame().T
    z = np.asarray(pre.transform(row_df))[0]

    out: list[tuple[str, object, float, float | None]] = []

    # Numeric block comes first, 1:1 with NUMERIC_FEATURES.
    for i, feature in enumerate(NUMERIC_FEATURES):
        out.append((feature, row[feature], float(coef[i] * z[i]), float(z[i])))

    # Categorical one-hot block: for each feature, only its active category
    # has encoded value 1, so its contribution is that column's coefficient.
    ohe = pre.named_transformers_["cat"]
    offset = len(NUMERIC_FEATURES)
    for j, feature in enumerate(CATEGORICAL_FEATURES):
        categories = list(ohe.categories_[j])
        value = row[feature]
        if value in categories:
            idx = offset + categories.index(value)
            out.append((feature, value, float(coef[idx] * z[idx]), None))
        offset += len(categories)

    return out


def case_from_row(
    model: TrainedRiskModel,
    row: pd.Series,
    case_id: str,
    member_id: str,
    top_k: int = 4,
    model_version: str = "2026.06",
) -> RiskCase:
    """Build a RiskCase from the model's prediction + attribution for one row."""
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    row_df = row[features].to_frame().T
    risk_score = float(model.pipeline.predict_proba(row_df)[0, 1])

    contribs = [
        c for c in _contributions(model, row) if abs(c[2]) > _MIN_ABS_CONTRIBUTION
    ]
    contribs.sort(key=lambda c: abs(c[2]), reverse=True)
    selected = contribs[:top_k] or contribs[:1]

    total_abs = sum(abs(c[2]) for c in selected) or 1.0
    factors = [
        RiskFactor(
            name=readable(feature),
            direction=(
                FactorDirection.INCREASES_RISK
                if contribution > 0
                else FactorDirection.DECREASES_RISK
            ),
            weight=round(min(1.0, abs(contribution) / total_abs), 4),
            evidence=_evidence(feature, value, z),
        )
        for feature, value, contribution, z in selected
    ]

    return RiskCase(
        member_id=member_id,
        case_id=case_id,
        risk_score=round(risk_score, 4),
        risk_band=_band(risk_score),
        model_name="german_credit_logreg",
        model_version=model_version,
        factors=factors,
    )
