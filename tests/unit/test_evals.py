"""Tests for the deterministic evaluation harness."""

from __future__ import annotations

from app.evals.coverage import evaluate_coverage
from app.evals.faithfulness import evaluate_faithfulness
from app.evals.readability import evaluate_readability
from app.evals.runner import evaluate_explanation
from app.evals.safety import evaluate_safety
from app.evals.text import flesch_kincaid_grade, overlap_ratio
from app.llm.clients import FakeLLMClient
from app.llm.prompts import build_user_prompt
from app.llm.structured_outputs import parse_explanation
from app.schemas.explanation import ExplanationOutput, FactorExplanation
from app.schemas.risk_case import RiskCase

CASE = RiskCase.model_validate(
    {
        "member_id": "mem_1",
        "case_id": "case_1",
        "risk_score": 0.8,
        "risk_band": "high",
        "model_name": "m",
        "model_version": "v1",
        "factors": [
            {
                "name": "Recent emergency visit",
                "direction": "increases_risk",
                "weight": 0.6,
                "evidence": "Member had two emergency visits in the last sixty days.",
            },
            {
                "name": "Medication refill gap",
                "direction": "increases_risk",
                "weight": 0.4,
                "evidence": "Refill gap of twenty one days detected.",
            },
        ],
    }
)


def _grounded_explanation() -> ExplanationOutput:
    raw = FakeLLMClient().complete(system="", user=build_user_prompt(CASE))
    return parse_explanation(raw)


# --- text utilities ---


def test_overlap_ratio_bounds() -> None:
    assert overlap_ratio("", "anything") == 1.0
    assert overlap_ratio("emergency visits", "emergency visits happened") == 1.0
    assert overlap_ratio("zebra giraffe", "emergency visits") == 0.0


def test_flesch_kincaid_simple_vs_complex() -> None:
    simple = flesch_kincaid_grade("The cat sat. The dog ran.")
    complex_text = flesch_kincaid_grade(
        "The multivariate logistic regression coefficient demonstrated "
        "statistically significant heteroskedastic propensity."
    )
    assert complex_text > simple


# --- faithfulness ---


def test_faithfulness_passes_for_grounded_output() -> None:
    res = evaluate_faithfulness(_grounded_explanation(), CASE)
    assert res.passed
    assert res.faithfulness_score >= 0.85
    assert res.unsupported_claims == []


def test_faithfulness_detects_unsupported_claim() -> None:
    bad = ExplanationOutput(
        summary="High risk.",
        member_explanation="The member has chronic diabetes and heart disease.",
        factor_explanations=[
            FactorExplanation(
                factor_name="Recent emergency visit",
                plain_english_reason="The member enjoys hiking and travel.",
                source_evidence_used=[],
            )
        ],
        limitations=[],
    )
    res = evaluate_faithfulness(bad, CASE)
    assert not res.passed
    assert res.unsupported_claims


def test_faithfulness_detects_fabricated_citation() -> None:
    bad = ExplanationOutput(
        summary="s",
        member_explanation="Member had two emergency visits in the last sixty days.",
        factor_explanations=[
            FactorExplanation(
                factor_name="Recent emergency visit",
                plain_english_reason="Member had two emergency visits.",
                source_evidence_used=["Totally made-up evidence not in the case."],
            )
        ],
        limitations=[],
    )
    res = evaluate_faithfulness(bad, CASE)
    assert not res.passed
    assert any("fabricated" in u for u in res.unsupported_claims)


# --- coverage ---


def test_coverage_full_for_grounded_output() -> None:
    res = evaluate_coverage(_grounded_explanation(), CASE)
    assert res.coverage_score == 1.0
    assert res.missing_factors == []
    assert res.passed


def test_coverage_penalizes_missing_high_weight_factor() -> None:
    # Only explains the low-weight factor (0.4 of 1.0 total).
    partial = ExplanationOutput(
        summary="Refill gap noted.",
        member_explanation="There was a refill gap of twenty one days detected.",
        factor_explanations=[
            FactorExplanation(
                factor_name="Medication refill gap",
                plain_english_reason="Refill gap of twenty one days detected.",
                source_evidence_used=["Refill gap of twenty one days detected."],
            )
        ],
        limitations=[],
    )
    res = evaluate_coverage(partial, CASE)
    assert "Recent emergency visit" in res.missing_factors
    assert abs(res.coverage_score - 0.4) < 1e-6
    assert not res.passed  # 0.4 < 0.75 threshold


# --- readability ---


def test_readability_passes_for_plain_language() -> None:
    plain = ExplanationOutput(
        summary="Your risk is high.",
        member_explanation="You went to the hospital twice. You missed some refills.",
        factor_explanations=[],
        limitations=[],
    )
    res = evaluate_readability(plain)
    assert res.passed
    assert res.jargon_terms == []


def test_readability_flags_jargon() -> None:
    technical = ExplanationOutput(
        summary="Risk computed.",
        member_explanation=(
            "The logistic regression coefficient and propensity covariate "
            "yielded a high posterior quantile."
        ),
        factor_explanations=[],
        limitations=[],
    )
    res = evaluate_readability(technical)
    assert not res.passed
    assert "regression" in res.jargon_terms


# --- safety ---


def test_safety_passes_for_compliant_output() -> None:
    assert evaluate_safety(_grounded_explanation()).passed


def test_safety_flags_medical_advice() -> None:
    bad = ExplanationOutput(
        summary="s",
        member_explanation="You should stop taking your medication immediately.",
        factor_explanations=[],
        limitations=[],
    )
    res = evaluate_safety(bad)
    assert not res.passed
    assert "medical_advice" in res.flags


def test_safety_flags_guarantee_and_blame() -> None:
    bad = ExplanationOutput(
        summary="We guarantee you will not be readmitted.",
        member_explanation="This is your fault for missing appointments.",
        factor_explanations=[],
        limitations=[],
    )
    res = evaluate_safety(bad)
    assert {"guarantee", "blame"}.issubset(set(res.flags))


# --- aggregate ---


def test_evaluate_explanation_aggregate_passes() -> None:
    result = evaluate_explanation(_grounded_explanation(), CASE)
    assert result.passed
    fields = result.to_record_fields()
    assert fields["decision"] == "pass"
    assert fields["safety_passed"] is True
    assert "failure_reasons" in fields["details"]


def test_aggregate_failure_reasons_listed() -> None:
    bad = ExplanationOutput(
        summary="We guarantee success.",
        member_explanation="You have diabetes because of your age.",
        factor_explanations=[],
        limitations=[],
    )
    result = evaluate_explanation(bad, CASE)
    assert not result.passed
    reasons = result.failure_reasons()
    assert "safety" in reasons
    assert "coverage" in reasons  # no factors explained
