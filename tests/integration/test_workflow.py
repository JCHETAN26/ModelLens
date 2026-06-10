"""LangGraph explanation workflow tests."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import repository
from app.db.models import AuditLogRecord, EvalResultRecord, ExplanationRecord
from app.graph.workflow import build_workflow
from app.llm.clients import FakeLLMClient
from app.schemas.explanation import ExplanationOutput, FactorExplanation
from app.schemas.risk_case import RiskCase

CASE_PAYLOAD = {
    "member_id": "mem_1",
    "case_id": "case_wf",
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


def _seed_case(session: Session) -> None:
    repository.create_risk_case(session, RiskCase.model_validate(CASE_PAYLOAD))
    session.commit()


class _StaticClient:
    """Always returns the same (valid JSON) response."""

    def __init__(self, output: ExplanationOutput) -> None:
        self._json = output.model_dump_json()

    def complete(self, *, system: str, user: str) -> str:
        return self._json


class _BadJsonClient:
    def complete(self, *, system: str, user: str) -> str:
        return "not json at all"


def test_workflow_passes_and_persists(db_session: Session) -> None:
    _seed_case(db_session)
    wf = build_workflow(db_session, llm_client=FakeLLMClient())
    final = wf.run("case_wf")

    assert final["status"] == "passed"
    assert final["explanation_id"] is not None
    assert final["eval_result"].passed

    explanation = db_session.scalar(select(ExplanationRecord))
    assert explanation is not None
    assert explanation.status == "passed"

    eval_row = db_session.scalar(select(EvalResultRecord))
    assert eval_row is not None
    assert eval_row.decision == "pass"

    # Every run produces an audit log.
    audit = db_session.scalar(select(AuditLogRecord))
    assert audit is not None
    assert audit.event == "explanation_passed"


def _unsafe_output() -> ExplanationOutput:
    # Grounded + full coverage, but trips the safety evaluator (guarantee).
    return ExplanationOutput(
        summary="We guarantee you will have no risk at all.",
        member_explanation=(
            "Member had two emergency visits in the last sixty days. "
            "Refill gap of twenty one days detected."
        ),
        factor_explanations=[
            FactorExplanation(
                factor_name="Recent emergency visit",
                plain_english_reason="Member had two emergency visits.",
                source_evidence_used=[
                    "Member had two emergency visits in the last sixty days."
                ],
            ),
            FactorExplanation(
                factor_name="Medication refill gap",
                plain_english_reason="Refill gap of twenty one days detected.",
                source_evidence_used=["Refill gap of twenty one days detected."],
            ),
        ],
        limitations=[],
    )


def test_workflow_retries_then_fails_after_max(db_session: Session) -> None:
    _seed_case(db_session)
    client = _StaticClient(_unsafe_output())
    wf = build_workflow(db_session, llm_client=client, max_retries=2)
    final = wf.run("case_wf")

    assert final["status"] == "failed"
    assert final["retry_count"] == 2  # exhausted the retry budget
    assert "safety" in final["eval_result"].failure_reasons()

    explanation = db_session.scalar(select(ExplanationRecord))
    assert explanation is not None
    assert explanation.status == "failed"

    audit = db_session.scalar(
        select(AuditLogRecord).where(AuditLogRecord.event == "explanation_failed")
    )
    assert audit is not None
    assert "safety" in audit.payload["reasons"]


def test_workflow_handles_generation_failure(db_session: Session) -> None:
    _seed_case(db_session)
    wf = build_workflow(db_session, llm_client=_BadJsonClient(), max_retries=2)
    final = wf.run("case_wf")

    assert final["status"] == "failed"
    audit = db_session.scalar(
        select(AuditLogRecord).where(AuditLogRecord.event == "explanation_failed")
    )
    assert audit is not None


def test_workflow_missing_case_fails_gracefully(db_session: Session) -> None:
    wf = build_workflow(db_session, llm_client=FakeLLMClient())
    final = wf.run("does_not_exist")
    assert final["status"] == "failed"
    # Audit log records the failure even with no case.
    audit = db_session.scalar(select(AuditLogRecord))
    assert audit is not None
    assert audit.event == "explanation_failed"
