"""Run all deterministic evaluators and aggregate the result."""

from __future__ import annotations

from app.evals.coverage import evaluate_coverage
from app.evals.faithfulness import evaluate_faithfulness
from app.evals.readability import evaluate_readability
from app.evals.safety import evaluate_safety
from app.schemas.eval import EvaluationResult
from app.schemas.explanation import ExplanationOutput
from app.schemas.risk_case import RiskCase


def evaluate_explanation(
    explanation: ExplanationOutput, case: RiskCase
) -> EvaluationResult:
    """Run faithfulness, coverage, readability, and safety evaluators."""
    return EvaluationResult(
        faithfulness=evaluate_faithfulness(explanation, case),
        coverage=evaluate_coverage(explanation, case),
        readability=evaluate_readability(explanation),
        safety=evaluate_safety(explanation),
    )
