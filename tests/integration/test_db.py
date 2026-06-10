"""Database layer tests: migrations apply, records persist and link back."""

from __future__ import annotations

from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from app.db.models import (
    EvalResultRecord,
    ExplanationRecord,
    Member,
    RiskCaseRecord,
    RiskFactorRecord,
)

EXPECTED_TABLES = {
    "members",
    "risk_cases",
    "risk_factors",
    "explanations",
    "eval_results",
    "audit_logs",
}


def test_migrations_create_all_tables(db_session: Session) -> None:
    tables = set(inspect(db_session.get_bind()).get_table_names())
    assert EXPECTED_TABLES.issubset(tables)


def test_insert_and_query_risk_case(db_session: Session) -> None:
    member = Member(member_id="mem_123")
    case = RiskCaseRecord(
        case_id="case_001",
        member=member,
        risk_score=0.87,
        risk_band="high",
        model_name="readmission_risk_v1",
        model_version="2026.01",
        factors=[
            RiskFactorRecord(
                name="Recent emergency visit",
                direction="increases_risk",
                weight=0.31,
                evidence="2 ER visits in the last 60 days.",
            )
        ],
    )
    db_session.add(case)
    db_session.commit()

    fetched = db_session.scalar(
        select(RiskCaseRecord).where(RiskCaseRecord.case_id == "case_001")
    )
    assert fetched is not None
    assert fetched.member.member_id == "mem_123"
    assert len(fetched.factors) == 1
    assert fetched.factors[0].name == "Recent emergency visit"


def test_explanation_and_eval_link_back_to_case(db_session: Session) -> None:
    member = Member(member_id="mem_456")
    case = RiskCaseRecord(
        case_id="case_002",
        member=member,
        risk_score=0.4,
        risk_band="medium",
        model_name="claim_risk_v1",
        model_version="2026.02",
        factors=[
            RiskFactorRecord(
                name="High claim amount",
                direction="increases_risk",
                weight=0.5,
                evidence="Claim of $42,000 exceeds the policy average.",
            )
        ],
    )
    explanation = ExplanationRecord(
        risk_case=case,
        summary="Moderate risk driven by claim amount.",
        member_explanation="The claim amount is higher than usual.",
        factor_explanations=[{"factor_name": "High claim amount"}],
        limitations=["Based only on provided data."],
        status="passed",
    )
    eval_result = EvalResultRecord(
        explanation=explanation,
        faithfulness_score=0.92,
        coverage_score=0.88,
        readability_score=9.0,
        safety_passed=True,
        decision="pass",
        details={"unsupported_claims": []},
    )
    db_session.add(eval_result)
    db_session.commit()

    fetched_eval = db_session.scalar(select(EvalResultRecord))
    assert fetched_eval is not None
    # eval → explanation → case → member, all traceable.
    assert fetched_eval.explanation.risk_case.case_id == "case_002"
    assert fetched_eval.explanation.risk_case.member.member_id == "mem_456"
    assert fetched_eval.decision == "pass"
