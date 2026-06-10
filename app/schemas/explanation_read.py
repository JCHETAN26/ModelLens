"""Read/response schemas for generated explanations and their evaluation."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.db.models import EvalResultRecord, ExplanationRecord
from app.schemas.explanation import FactorExplanation


class EvalSummary(BaseModel):
    faithfulness_score: float
    coverage_score: float
    readability_grade: float
    safety_passed: bool
    decision: str
    details: dict


class ExplanationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    case_id: str
    status: str
    retry_count: int
    summary: str
    member_explanation: str
    factor_explanations: list[FactorExplanation]
    limitations: list[str]
    created_at: datetime
    evaluation: EvalSummary | None


def _eval_summary(record: EvalResultRecord | None) -> EvalSummary | None:
    if record is None:
        return None
    return EvalSummary(
        faithfulness_score=record.faithfulness_score,
        coverage_score=record.coverage_score,
        readability_grade=record.readability_score,
        safety_passed=record.safety_passed,
        decision=record.decision,
        details=record.details,
    )


def explanation_to_read(record: ExplanationRecord, case_id: str) -> ExplanationRead:
    """Assemble the API response from an explanation record and its latest eval."""
    latest_eval = (
        max(record.eval_results, key=lambda e: e.id) if record.eval_results else None
    )
    return ExplanationRead(
        id=record.id,
        case_id=case_id,
        status=record.status,
        retry_count=record.retry_count,
        summary=record.summary,
        member_explanation=record.member_explanation,
        factor_explanations=[
            FactorExplanation.model_validate(fe) for fe in record.factor_explanations
        ],
        limitations=list(record.limitations),
        created_at=record.created_at,
        evaluation=_eval_summary(latest_eval),
    )
