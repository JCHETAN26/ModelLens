"""Explanation generation endpoints (build-plan §10, Task 3.2)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session

from app.core.errors import (
    CaseNotFoundError,
    ExplanationGenerationError,
    ExplanationNotFoundError,
)
from app.db import repository
from app.db.session import get_db
from app.graph.workflow import build_workflow
from app.schemas.explanation_read import ExplanationRead, explanation_to_read

router = APIRouter(prefix="/cases/{case_id}/explanations", tags=["explanations"])

DbSession = Annotated[Session, Depends(get_db)]


@router.post("", response_model=ExplanationRead, status_code=status.HTTP_201_CREATED)
def create_explanation(
    case_id: str, db: DbSession, response: Response
) -> ExplanationRead:
    """Trigger the LangGraph workflow to generate and evaluate an explanation.

    A failed-evaluation outcome is a valid result (status="failed"), not a
    server error — the caller gets the reasons in the eval details.
    """
    if repository.get_risk_case(db, case_id) is None:
        raise CaseNotFoundError(f"case_id '{case_id}' not found")

    workflow = build_workflow(db)
    final = workflow.run(case_id)

    explanation_id = final.get("explanation_id")
    if explanation_id is None:
        raise ExplanationGenerationError(
            f"explanation could not be generated for '{case_id}'"
        )

    record = repository.get_explanation(db, explanation_id)
    if record is None:  # pragma: no cover - defensive
        raise ExplanationGenerationError("explanation record missing after run")

    if final.get("status") != "passed":
        # Record created, but evaluation did not pass; surface 200 not 201.
        response.status_code = status.HTTP_200_OK
    return explanation_to_read(record, case_id)


@router.get("/latest", response_model=ExplanationRead)
def get_latest_explanation(case_id: str, db: DbSession) -> ExplanationRead:
    if repository.get_risk_case(db, case_id) is None:
        raise CaseNotFoundError(f"case_id '{case_id}' not found")
    record = repository.get_latest_explanation(db, case_id)
    if record is None:
        raise ExplanationNotFoundError(
            f"no explanation found for case_id '{case_id}'"
        )
    return explanation_to_read(record, case_id)
