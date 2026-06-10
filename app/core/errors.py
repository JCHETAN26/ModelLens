"""Domain-specific exceptions and FastAPI exception handlers."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse


class ModelLensError(Exception):
    """Base class for all ModelLens domain errors."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class CaseNotFoundError(ModelLensError):
    status_code = 404
    code = "case_not_found"


class ExplanationGenerationError(ModelLensError):
    status_code = 502
    code = "explanation_generation_failed"


class LLMOutputValidationError(ModelLensError):
    status_code = 502
    code = "llm_output_invalid"


async def modellens_error_handler(_: Request, exc: ModelLensError) -> JSONResponse:
    """Render ModelLens domain errors as structured JSON."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )
