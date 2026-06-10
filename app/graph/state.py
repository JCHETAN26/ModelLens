"""LangGraph workflow state and execution context (build-plan §12.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from sqlalchemy.orm import Session

from app.llm.clients import LLMClient
from app.schemas.eval import (
    CoverageResult,
    EvaluationResult,
    FaithfulnessResult,
    ReadabilityResult,
    SafetyResult,
)
from app.schemas.explanation import ExplanationOutput
from app.schemas.risk_case import RiskCase


class GraphState(TypedDict, total=False):
    """Mutable state threaded through the explanation workflow."""

    case_id: str
    risk_case: RiskCase | None
    draft_explanation: ExplanationOutput | None

    # Individual evaluator outputs, set by the four evaluate_* nodes.
    faithfulness: FaithfulnessResult | None
    coverage: CoverageResult | None
    readability: ReadabilityResult | None
    safety: SafetyResult | None
    eval_result: EvaluationResult | None

    retry_count: int
    final_explanation: ExplanationOutput | None
    explanation_id: int | None
    status: str
    errors: list[str]


@dataclass
class WorkflowContext:
    """Dependencies bound into the graph for a single run."""

    llm_client: LLMClient
    session: Session
    max_retries: int
