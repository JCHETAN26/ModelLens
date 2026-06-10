"""Workflow node implementations (build-plan §12.2).

Each node takes the current state and the run context and returns a partial
state update (LangGraph merges it). Nodes are deliberately small and pure-ish:
the only side effects are LLM calls (generate/rewrite) and DB writes
(persist/fail), each isolated to its own node.
"""

from __future__ import annotations

from app.core.errors import LLMOutputValidationError
from app.core.logging import get_logger
from app.db import repository
from app.db.models import AuditLogRecord, EvalResultRecord, ExplanationRecord
from app.evals.coverage import evaluate_coverage
from app.evals.faithfulness import evaluate_faithfulness
from app.evals.readability import evaluate_readability
from app.evals.safety import evaluate_safety
from app.graph.state import GraphState, WorkflowContext
from app.llm.structured_outputs import generate_explanation as llm_generate
from app.schemas.eval import EvaluationResult

logger = get_logger(__name__)


def load_case(state: GraphState, ctx: WorkflowContext) -> dict:
    """Ensure the risk case is available, loading from DB if only an id is given."""
    update: dict = {"retry_count": 0, "errors": [], "status": "loading"}
    case = state.get("risk_case")
    if case is None:
        record = repository.get_risk_case(ctx.session, state["case_id"])
        if record is None:
            return {
                "status": "failed",
                "errors": [f"case '{state['case_id']}' not found"],
                "risk_case": None,
            }
        case = repository.record_to_schema(record)
    update["risk_case"] = case
    update["case_id"] = case.case_id
    return update


def generate_explanation(state: GraphState, ctx: WorkflowContext) -> dict:
    """Generate the first draft explanation via the LLM (with internal retry)."""
    case = state.get("risk_case")
    if case is None:
        return {"status": "failed", "draft_explanation": None}
    try:
        draft = llm_generate(case, ctx.llm_client)
    except LLMOutputValidationError as exc:
        logger.warning("generation failed for %s: %s", state.get("case_id"), exc)
        return {
            "status": "generation_failed",
            "draft_explanation": None,
            "errors": [*state.get("errors", []), f"generation: {exc}"],
        }
    return {"draft_explanation": draft, "status": "evaluating"}


def evaluate_faithfulness_node(state: GraphState, ctx: WorkflowContext) -> dict:
    draft = state["draft_explanation"]
    case = state["risk_case"]
    return {"faithfulness": evaluate_faithfulness(draft, case)}


def evaluate_coverage_node(state: GraphState, ctx: WorkflowContext) -> dict:
    draft = state["draft_explanation"]
    case = state["risk_case"]
    return {"coverage": evaluate_coverage(draft, case)}


def evaluate_readability_node(state: GraphState, ctx: WorkflowContext) -> dict:
    return {"readability": evaluate_readability(state["draft_explanation"])}


def evaluate_safety_node(state: GraphState, ctx: WorkflowContext) -> dict:
    safety = evaluate_safety(state["draft_explanation"])
    eval_result = EvaluationResult(
        faithfulness=state["faithfulness"],
        coverage=state["coverage"],
        readability=state["readability"],
        safety=safety,
    )
    status = "passed" if eval_result.passed else "needs_rewrite"
    return {"safety": safety, "eval_result": eval_result, "status": status}


def rewrite_explanation(state: GraphState, ctx: WorkflowContext) -> dict:
    """Increment the retry counter and regenerate the explanation."""
    retry_count = state.get("retry_count", 0) + 1
    case = state["risk_case"]
    try:
        draft = llm_generate(case, ctx.llm_client)
    except LLMOutputValidationError as exc:
        return {
            "retry_count": retry_count,
            "status": "generation_failed",
            "draft_explanation": None,
            "errors": [*state.get("errors", []), f"rewrite: {exc}"],
        }
    return {
        "retry_count": retry_count,
        "draft_explanation": draft,
        "status": "rewriting",
    }


def _audit(ctx: WorkflowContext, case_id: str, explanation_id: int | None, event: str, payload: dict) -> None:
    ctx.session.add(
        AuditLogRecord(
            case_id=case_id,
            explanation_id=explanation_id,
            event=event,
            payload=payload,
        )
    )


def persist_result(state: GraphState, ctx: WorkflowContext) -> dict:
    """Persist a passing explanation, its eval result, and an audit log."""
    case_id = state["case_id"]
    record = repository.get_risk_case(ctx.session, case_id)
    draft = state["draft_explanation"]
    eval_result: EvaluationResult = state["eval_result"]

    explanation = ExplanationRecord(
        risk_case_id=record.id,
        summary=draft.summary,
        member_explanation=draft.member_explanation,
        factor_explanations=[fe.model_dump() for fe in draft.factor_explanations],
        limitations=draft.limitations,
        status="passed",
        retry_count=state.get("retry_count", 0),
    )
    ctx.session.add(explanation)
    ctx.session.flush()

    ctx.session.add(
        EvalResultRecord(
            explanation_id=explanation.id, **eval_result.to_record_fields()
        )
    )
    _audit(
        ctx,
        case_id,
        explanation.id,
        "explanation_passed",
        {"retry_count": state.get("retry_count", 0)},
    )
    ctx.session.flush()
    return {
        "final_explanation": draft,
        "explanation_id": explanation.id,
        "status": "passed",
    }


def fail_gracefully(state: GraphState, ctx: WorkflowContext) -> dict:
    """Persist a failed attempt with its reasons and an audit log."""
    case_id = state.get("case_id", "")
    record = repository.get_risk_case(ctx.session, case_id)
    draft = state.get("draft_explanation")
    eval_result: EvaluationResult | None = state.get("eval_result")

    explanation_id: int | None = None
    if record is not None:
        explanation = ExplanationRecord(
            risk_case_id=record.id,
            summary=draft.summary if draft else "",
            member_explanation=draft.member_explanation if draft else "",
            factor_explanations=(
                [fe.model_dump() for fe in draft.factor_explanations] if draft else []
            ),
            limitations=draft.limitations if draft else [],
            status="failed",
            retry_count=state.get("retry_count", 0),
        )
        ctx.session.add(explanation)
        ctx.session.flush()
        explanation_id = explanation.id
        if eval_result is not None:
            ctx.session.add(
                EvalResultRecord(
                    explanation_id=explanation.id, **eval_result.to_record_fields()
                )
            )

    reasons = eval_result.failure_reasons() if eval_result else state.get("errors", [])
    _audit(
        ctx,
        case_id,
        explanation_id,
        "explanation_failed",
        {"retry_count": state.get("retry_count", 0), "reasons": reasons},
    )
    ctx.session.flush()
    return {"status": "failed", "explanation_id": explanation_id}
