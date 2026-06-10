"""Evaluation result schemas.

Each deterministic evaluator returns its own typed result; `EvaluationResult`
aggregates them and flattens into the `eval_results` table columns.
"""

from __future__ import annotations

from pydantic import BaseModel


class FaithfulnessResult(BaseModel):
    faithfulness_score: float
    supported_claims: list[str]
    unsupported_claims: list[str]
    passed: bool

    @property
    def decision(self) -> str:
        return "pass" if self.passed else "fail"


class CoverageResult(BaseModel):
    coverage_score: float
    covered_factors: list[str]
    missing_factors: list[str]
    passed: bool


class ReadabilityResult(BaseModel):
    # Flesch-Kincaid grade level (lower = more readable).
    grade_level: float
    avg_sentence_length: float
    jargon_terms: list[str]
    passed: bool


class SafetyResult(BaseModel):
    passed: bool
    flags: list[str]


class EvaluationResult(BaseModel):
    faithfulness: FaithfulnessResult
    coverage: CoverageResult
    readability: ReadabilityResult
    safety: SafetyResult

    @property
    def passed(self) -> bool:
        return (
            self.faithfulness.passed
            and self.coverage.passed
            and self.readability.passed
            and self.safety.passed
        )

    def failure_reasons(self) -> list[str]:
        reasons: list[str] = []
        if not self.faithfulness.passed:
            reasons.append("faithfulness")
        if not self.coverage.passed:
            reasons.append("coverage")
        if not self.readability.passed:
            reasons.append("readability")
        if not self.safety.passed:
            reasons.append("safety")
        return reasons

    def to_record_fields(self) -> dict:
        """Flatten into columns for the `eval_results` table."""
        return {
            "faithfulness_score": round(self.faithfulness.faithfulness_score, 4),
            "coverage_score": round(self.coverage.coverage_score, 4),
            "readability_score": round(self.readability.grade_level, 2),
            "safety_passed": self.safety.passed,
            "decision": "pass" if self.passed else "fail",
            "details": {
                "unsupported_claims": self.faithfulness.unsupported_claims,
                "missing_factors": self.coverage.missing_factors,
                "jargon_terms": self.readability.jargon_terms,
                "safety_flags": self.safety.flags,
                "failure_reasons": self.failure_reasons(),
            },
        }
