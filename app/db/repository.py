"""Persistence helpers for risk cases.

Thin functions over the ORM so routes stay declarative and the same logic is
reusable from the LangGraph workflow and batch eval CLI.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.errors import CaseAlreadyExistsError
from app.db.models import Member, RiskCaseRecord, RiskFactorRecord
from app.schemas.risk_case import RiskCase


def _get_or_create_member(session: Session, member_id: str) -> Member:
    member = session.scalar(select(Member).where(Member.member_id == member_id))
    if member is None:
        member = Member(member_id=member_id)
        session.add(member)
        session.flush()
    return member


def create_risk_case(session: Session, payload: RiskCase) -> RiskCaseRecord:
    """Persist a validated risk case. Raises if `case_id` already exists."""
    existing = session.scalar(
        select(RiskCaseRecord).where(RiskCaseRecord.case_id == payload.case_id)
    )
    if existing is not None:
        raise CaseAlreadyExistsError(f"case_id '{payload.case_id}' already exists")

    member = _get_or_create_member(session, payload.member_id)
    record = RiskCaseRecord(
        case_id=payload.case_id,
        member=member,
        risk_score=payload.risk_score,
        risk_band=payload.risk_band.value,
        model_name=payload.model_name,
        model_version=payload.model_version,
        factors=[
            RiskFactorRecord(
                name=f.name,
                direction=f.direction.value,
                weight=f.weight,
                evidence=f.evidence,
            )
            for f in payload.factors
        ],
    )
    session.add(record)
    session.flush()
    return record


def get_risk_case(session: Session, case_id: str) -> RiskCaseRecord | None:
    """Fetch a risk case by its business `case_id`, eager-loading relations."""
    return session.scalar(
        select(RiskCaseRecord)
        .where(RiskCaseRecord.case_id == case_id)
        .options(
            selectinload(RiskCaseRecord.factors),
            selectinload(RiskCaseRecord.member),
        )
    )


def list_risk_cases(
    session: Session, limit: int = 50, offset: int = 0
) -> list[RiskCaseRecord]:
    """List risk cases, newest first."""
    return list(
        session.scalars(
            select(RiskCaseRecord)
            .options(
                selectinload(RiskCaseRecord.factors),
                selectinload(RiskCaseRecord.member),
            )
            .order_by(RiskCaseRecord.created_at.desc(), RiskCaseRecord.id.desc())
            .limit(limit)
            .offset(offset)
        )
    )
