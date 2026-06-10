"""LangGraph explanation workflow assembly (build-plan §12).

Flow:
    load_case → generate_explanation → evaluate_faithfulness → evaluate_coverage
    → evaluate_readability → evaluate_safety
        ├─ all pass            → persist_result → END
        ├─ fail, retries left  → rewrite_explanation → evaluate_faithfulness ...
        └─ fail, retries spent → fail_gracefully → END

Generation failures short-circuit to fail_gracefully. Every run ends in either
persist_result or fail_gracefully, both of which write an audit log.
"""

from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.graph import nodes
from app.graph.state import GraphState, WorkflowContext
from app.llm.clients import LLMClient, get_llm_client
from app.schemas.risk_case import RiskCase


def _route_after_load(state: GraphState) -> str:
    return "fail" if state.get("status") == "failed" else "generate"


def _route_after_generate(state: GraphState) -> str:
    return "evaluate" if state.get("draft_explanation") is not None else "fail"


def _route_after_eval(state: GraphState, max_retries: int) -> str:
    eval_result = state["eval_result"]
    if eval_result.passed:
        return "persist"
    if state.get("retry_count", 0) < max_retries:
        return "rewrite"
    return "fail"


def _route_after_rewrite(state: GraphState) -> str:
    return "evaluate" if state.get("draft_explanation") is not None else "fail"


class ExplanationWorkflow:
    """Compiled, dependency-bound explanation workflow."""

    def __init__(self, ctx: WorkflowContext) -> None:
        self.ctx = ctx
        self.graph = self._build()

    def _bind(self, fn) -> Callable[[GraphState], dict]:
        ctx = self.ctx

        def node(state: GraphState) -> dict:
            return fn(state, ctx)

        return node

    def _build(self):
        g = StateGraph(GraphState)

        g.add_node("load_case", self._bind(nodes.load_case))
        g.add_node("generate_explanation", self._bind(nodes.generate_explanation))
        g.add_node("evaluate_faithfulness", self._bind(nodes.evaluate_faithfulness_node))
        g.add_node("evaluate_coverage", self._bind(nodes.evaluate_coverage_node))
        g.add_node("evaluate_readability", self._bind(nodes.evaluate_readability_node))
        g.add_node("evaluate_safety", self._bind(nodes.evaluate_safety_node))
        g.add_node("rewrite_explanation", self._bind(nodes.rewrite_explanation))
        g.add_node("persist_result", self._bind(nodes.persist_result))
        g.add_node("fail_gracefully", self._bind(nodes.fail_gracefully))

        g.set_entry_point("load_case")
        g.add_conditional_edges(
            "load_case",
            _route_after_load,
            {"generate": "generate_explanation", "fail": "fail_gracefully"},
        )
        g.add_conditional_edges(
            "generate_explanation",
            _route_after_generate,
            {"evaluate": "evaluate_faithfulness", "fail": "fail_gracefully"},
        )
        g.add_edge("evaluate_faithfulness", "evaluate_coverage")
        g.add_edge("evaluate_coverage", "evaluate_readability")
        g.add_edge("evaluate_readability", "evaluate_safety")
        max_retries = self.ctx.max_retries
        g.add_conditional_edges(
            "evaluate_safety",
            lambda state: _route_after_eval(state, max_retries),
            {
                "persist": "persist_result",
                "rewrite": "rewrite_explanation",
                "fail": "fail_gracefully",
            },
        )
        g.add_conditional_edges(
            "rewrite_explanation",
            _route_after_rewrite,
            {"evaluate": "evaluate_faithfulness", "fail": "fail_gracefully"},
        )
        g.add_edge("persist_result", END)
        g.add_edge("fail_gracefully", END)

        return g.compile()

    def run(self, case_id: str, risk_case: RiskCase | None = None) -> GraphState:
        """Execute the workflow for one case. Returns the final state."""
        initial: GraphState = {"case_id": case_id, "risk_case": risk_case}
        return self.graph.invoke(initial)  # type: ignore[return-value]


def build_workflow(
    session: Session,
    llm_client: LLMClient | None = None,
    max_retries: int | None = None,
) -> ExplanationWorkflow:
    """Construct a workflow bound to a DB session and LLM client."""
    settings = get_settings()
    ctx = WorkflowContext(
        llm_client=llm_client or get_llm_client(settings),
        session=session,
        max_retries=max_retries if max_retries is not None else settings.max_rewrite_retries,
    )
    return ExplanationWorkflow(ctx)
