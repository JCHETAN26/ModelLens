"""SQLAlchemy ORM models for ModelLens persistence.

Schema (build-plan §9):
    members → risk_cases → risk_factors
                        ↘ explanations → eval_results
    audit_logs reference a case and/or explanation for traceability.

Every generated artifact links back to its source case so explanations and
evaluations are always traceable to the structured input.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )


class Member(TimestampMixin, Base):
    __tablename__ = "members"

    id: Mapped[int] = mapped_column(primary_key=True)
    member_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)

    risk_cases: Mapped[list[RiskCaseRecord]] = relationship(
        back_populates="member", cascade="all, delete-orphan"
    )


class RiskCaseRecord(TimestampMixin, Base):
    __tablename__ = "risk_cases"

    id: Mapped[int] = mapped_column(primary_key=True)
    case_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    member_pk: Mapped[int] = mapped_column(ForeignKey("members.id"), index=True)

    risk_score: Mapped[float] = mapped_column(Float)
    risk_band: Mapped[str] = mapped_column(String(16))
    model_name: Mapped[str] = mapped_column(String(128))
    model_version: Mapped[str] = mapped_column(String(64))

    member: Mapped[Member] = relationship(back_populates="risk_cases")
    factors: Mapped[list[RiskFactorRecord]] = relationship(
        back_populates="risk_case", cascade="all, delete-orphan"
    )
    explanations: Mapped[list[ExplanationRecord]] = relationship(
        back_populates="risk_case", cascade="all, delete-orphan"
    )

    @property
    def member_id(self) -> str:
        """Flat accessor for the business member identifier (for read schemas)."""
        return self.member.member_id


class RiskFactorRecord(TimestampMixin, Base):
    __tablename__ = "risk_factors"

    id: Mapped[int] = mapped_column(primary_key=True)
    risk_case_id: Mapped[int] = mapped_column(
        ForeignKey("risk_cases.id"), index=True
    )
    name: Mapped[str] = mapped_column(String(256))
    direction: Mapped[str] = mapped_column(String(32))
    weight: Mapped[float] = mapped_column(Float)
    evidence: Mapped[str] = mapped_column(Text)

    risk_case: Mapped[RiskCaseRecord] = relationship(back_populates="factors")


class ExplanationRecord(TimestampMixin, Base):
    __tablename__ = "explanations"

    id: Mapped[int] = mapped_column(primary_key=True)
    risk_case_id: Mapped[int] = mapped_column(
        ForeignKey("risk_cases.id"), index=True
    )

    summary: Mapped[str] = mapped_column(Text, default="")
    member_explanation: Mapped[str] = mapped_column(Text, default="")
    # Structured arrays from the LLM contract, stored as JSON.
    factor_explanations: Mapped[list] = mapped_column(JSON, default=list)
    limitations: Mapped[list] = mapped_column(JSON, default=list)

    # Lifecycle: pending → passed | failed
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    retry_count: Mapped[int] = mapped_column(default=0)

    risk_case: Mapped[RiskCaseRecord] = relationship(back_populates="explanations")
    eval_results: Mapped[list[EvalResultRecord]] = relationship(
        back_populates="explanation", cascade="all, delete-orphan"
    )


class EvalResultRecord(TimestampMixin, Base):
    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    explanation_id: Mapped[int] = mapped_column(
        ForeignKey("explanations.id"), index=True
    )

    faithfulness_score: Mapped[float] = mapped_column(Float, default=0.0)
    coverage_score: Mapped[float] = mapped_column(Float, default=0.0)
    readability_score: Mapped[float] = mapped_column(Float, default=0.0)
    safety_passed: Mapped[bool] = mapped_column(Boolean, default=False)
    decision: Mapped[str] = mapped_column(String(16), default="fail")
    details: Mapped[dict] = mapped_column(JSON, default=dict)

    explanation: Mapped[ExplanationRecord] = relationship(
        back_populates="eval_results"
    )


class AuditLogRecord(TimestampMixin, Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    case_id: Mapped[str | None] = mapped_column(String(128), index=True)
    explanation_id: Mapped[int | None] = mapped_column(
        ForeignKey("explanations.id")
    )
    event: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
