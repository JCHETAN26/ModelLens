"""Tests for the real credit-risk model and the RiskCase adapter.

Requires the `ml` extra (scikit-learn, pandas).
"""

from __future__ import annotations

import pytest

from app.evals.runner import evaluate_explanation
from app.llm.clients import FakeLLMClient
from app.llm.structured_outputs import generate_explanation
from app.schemas.risk_case import RiskCase

pytest.importorskip("sklearn")
pytest.importorskip("pandas")

from app.risk_model.adapter import case_from_row  # noqa: E402
from app.risk_model.dataset import TARGET, load_german_credit  # noqa: E402
from app.risk_model.generate import generate_cases  # noqa: E402
from app.risk_model.train import train_risk_model  # noqa: E402


def test_dataset_loads() -> None:
    frame = load_german_credit()
    assert frame.shape == (1000, 21)
    assert TARGET in frame.columns
    assert set(frame[TARGET].unique()) == {"good", "bad"}


def test_model_trains_with_reasonable_metrics() -> None:
    model = train_risk_model(seed=42)
    # German Credit logistic regression lands around AUC ~0.80.
    assert model.auc > 0.74
    assert model.accuracy > 0.70
    assert model.n_train + model.n_test == 1000


def test_training_is_deterministic() -> None:
    a = train_risk_model(seed=42)
    b = train_risk_model(seed=42)
    assert a.auc == b.auc
    assert a.accuracy == b.accuracy


def test_case_from_row_is_valid_risk_case() -> None:
    model = train_risk_model(seed=42)
    row = model.test_frame.iloc[0]
    case = case_from_row(model, row, case_id="c1", member_id="m1", top_k=4)

    assert isinstance(case, RiskCase)
    assert 0.0 <= case.risk_score <= 1.0
    assert 1 <= len(case.factors) <= 4
    for factor in case.factors:
        assert 0.0 < factor.weight <= 1.0
        assert factor.evidence.strip()
    # Weights are normalized shares of the selected contributions.
    assert sum(f.weight for f in case.factors) <= 1.0001


def test_band_matches_score() -> None:
    model = train_risk_model(seed=42)
    for i in range(min(20, len(model.test_frame))):
        case = case_from_row(
            model, model.test_frame.iloc[i], case_id=f"c{i}", member_id=f"m{i}"
        )
        if case.risk_score >= 0.66:
            assert case.risk_band.value == "high"
        elif case.risk_score >= 0.33:
            assert case.risk_band.value == "medium"
        else:
            assert case.risk_band.value == "low"


def test_generated_real_cases_flow_through_pipeline() -> None:
    cases, metrics = generate_cases(n=10, seed=42)
    assert metrics["roc_auc"] > 0.74
    assert len(cases) == 10

    client = FakeLLMClient()
    for raw in cases:
        case = RiskCase.model_validate(raw)  # real cases satisfy the input schema
        explanation = generate_explanation(case, client)
        result = evaluate_explanation(explanation, case)
        # Evidence is echoed, so real cases are grounded and fully covered.
        assert result.faithfulness.passed
        assert result.coverage.passed
        assert result.safety.passed
