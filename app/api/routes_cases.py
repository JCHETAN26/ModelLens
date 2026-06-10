"""Risk case CRUD endpoints (build-plan §10, Task 3.1)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.core.errors import CaseNotFoundError
from app.db import repository
from app.db.session import get_db
from app.schemas.risk_case import RiskCase
from app.schemas.risk_case_read import RiskCaseRead

router = APIRouter(prefix="/cases", tags=["cases"])

DbSession = Annotated[Session, Depends(get_db)]


@router.post("", response_model=RiskCaseRead, status_code=status.HTTP_201_CREATED)
def create_case(payload: RiskCase, db: DbSession) -> RiskCaseRead:
    """Validate and persist a structured risk case."""
    record = repository.create_risk_case(db, payload)
    db.flush()
    return RiskCaseRead.model_validate(record)


@router.get("/{case_id}", response_model=RiskCaseRead)
def get_case(case_id: str, db: DbSession) -> RiskCaseRead:
    record = repository.get_risk_case(db, case_id)
    if record is None:
        raise CaseNotFoundError(f"case_id '{case_id}' not found")
    return RiskCaseRead.model_validate(record)


@router.get("", response_model=list[RiskCaseRead])
def list_cases(
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[RiskCaseRead]:
    records = repository.list_risk_cases(db, limit=limit, offset=offset)
    return [RiskCaseRead.model_validate(r) for r in records]
