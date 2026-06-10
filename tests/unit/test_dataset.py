"""Tests for the synthetic dataset generator and the committed dataset file."""

from __future__ import annotations

import json

from app.evals.dataset import (
    CASES_PATH,
    EXPECTED_PATH,
    build_dataset,
)
from app.schemas.risk_case import RiskCase

# Fields that must never appear (no PII/PHI leakage).
_FORBIDDEN_KEYS = {"ssn", "address", "dob", "name", "email", "phone"}


def test_generator_is_deterministic() -> None:
    a_cases, a_expected = build_dataset(seed=7)
    b_cases, b_expected = build_dataset(seed=7)
    assert a_cases == b_cases
    assert a_expected == b_expected


def test_dataset_has_at_least_50_cases() -> None:
    cases, _ = build_dataset()
    assert len(cases) >= 50


def test_all_generated_cases_validate() -> None:
    cases, expected = build_dataset()
    case_ids = set()
    for raw in cases:
        case = RiskCase.model_validate(raw)  # raises if invalid
        case_ids.add(case.case_id)
        assert expected[case.case_id]["expected_important_factors"]
        # No forbidden top-level keys.
        assert not (_FORBIDDEN_KEYS & set(raw.keys()))
    assert len(case_ids) == len(cases)  # unique case_ids


def test_covers_all_five_categories() -> None:
    _, expected = build_dataset()
    categories = {v["category"] for v in expected.values()}
    assert categories == {
        "healthcare_readmission",
        "insurance_claim",
        "credit_risk",
        "fraud_risk",
        "operational_risk",
    }


def test_committed_files_match_generator() -> None:
    """The checked-in dataset must equal a fresh generation (reproducible)."""
    cases, expected = build_dataset()
    on_disk_cases = json.loads(CASES_PATH.read_text())
    on_disk_expected = json.loads(EXPECTED_PATH.read_text())
    assert on_disk_cases == cases
    assert on_disk_expected == expected
