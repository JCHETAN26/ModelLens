"""Validation tests for the structured risk case input schema."""

import pytest
from pydantic import ValidationError

from app.schemas.risk_case import FactorDirection, RiskCase, RiskFactor


def _valid_case() -> dict:
    return {
        "member_id": "mem_123",
        "case_id": "case_001",
        "risk_score": 0.87,
        "risk_band": "high",
        "model_name": "readmission_risk_v1",
        "model_version": "2026.01",
        "factors": [
            {
                "name": "Recent emergency visit",
                "direction": "increases_risk",
                "weight": 0.31,
                "evidence": "Member had 2 ER visits in the last 60 days.",
            },
            {
                "name": "Medication refill gap",
                "direction": "increases_risk",
                "weight": 0.24,
                "evidence": "Refill gap of 21 days detected for prescribed medication.",
            },
        ],
    }


def test_valid_case_parses() -> None:
    case = RiskCase.model_validate(_valid_case())
    assert case.case_id == "case_001"
    assert case.risk_band.value == "high"
    assert len(case.factors) == 2
    assert case.factors[0].direction is FactorDirection.INCREASES_RISK


def test_top_factors_sorted_by_weight() -> None:
    case = RiskCase.model_validate(_valid_case())
    top = case.top_factors(1)
    assert len(top) == 1
    assert top[0].name == "Recent emergency visit"  # weight 0.31 > 0.24


@pytest.mark.parametrize("missing", ["member_id", "case_id", "risk_score", "factors"])
def test_missing_required_field_rejected(missing: str) -> None:
    payload = _valid_case()
    del payload[missing]
    with pytest.raises(ValidationError):
        RiskCase.model_validate(payload)


@pytest.mark.parametrize("bad_score", [-0.1, 1.1, 2.0])
def test_risk_score_out_of_range_rejected(bad_score: float) -> None:
    payload = _valid_case()
    payload["risk_score"] = bad_score
    with pytest.raises(ValidationError):
        RiskCase.model_validate(payload)


def test_empty_factors_rejected() -> None:
    payload = _valid_case()
    payload["factors"] = []
    with pytest.raises(ValidationError):
        RiskCase.model_validate(payload)


def test_factor_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        RiskFactor.model_validate(
            {"name": "x", "direction": "increases_risk", "weight": 0.5}
        )


def test_blank_evidence_rejected() -> None:
    with pytest.raises(ValidationError):
        RiskFactor.model_validate(
            {
                "name": "x",
                "direction": "increases_risk",
                "weight": 0.5,
                "evidence": "   ",
            }
        )


def test_invalid_direction_rejected() -> None:
    payload = _valid_case()
    payload["factors"][0]["direction"] = "makes_it_worse"
    with pytest.raises(ValidationError):
        RiskCase.model_validate(payload)


def test_invalid_risk_band_rejected() -> None:
    payload = _valid_case()
    payload["risk_band"] = "extreme"
    with pytest.raises(ValidationError):
        RiskCase.model_validate(payload)


def test_duplicate_factor_names_rejected() -> None:
    payload = _valid_case()
    payload["factors"][1]["name"] = "Recent emergency visit"
    with pytest.raises(ValidationError):
        RiskCase.model_validate(payload)


def test_unknown_extra_field_rejected() -> None:
    payload = _valid_case()
    payload["ssn"] = "000-00-0000"  # PII must never sneak in
    with pytest.raises(ValidationError):
        RiskCase.model_validate(payload)
