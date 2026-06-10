"""Batch evaluation CLI (build-plan §15).

Runs every case in a dataset through generation + the deterministic evaluators
and reports reproducible aggregate metrics.

    python -m app.evals.run_batch --input sample_data/risk_cases.json

Uses the configured LLM provider (default `fake`, so it runs offline and the
numbers are reproducible). Results are written to `eval_results.json`.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from app.core.errors import LLMOutputValidationError
from app.evals.runner import evaluate_explanation
from app.llm.clients import LLMClient, get_llm_client
from app.llm.structured_outputs import generate_explanation
from app.schemas.eval import EvaluationResult
from app.schemas.risk_case import RiskCase


class CaseOutcome(BaseModel):
    case_id: str
    passed: bool
    generated: bool
    faithfulness_score: float
    coverage_score: float
    readability_grade: float
    safety_passed: bool
    unsupported_claim_count: int
    expected_factor_coverage: float | None
    failure_reasons: list[str]
    latency_ms: float


class BatchReport(BaseModel):
    total_cases: int
    generated: int
    passed: int
    pass_rate: float
    avg_faithfulness: float
    avg_coverage: float
    avg_readability_grade: float
    avg_latency_ms: float
    unsupported_claim_rate: float
    avg_expected_factor_coverage: float | None
    failure_reason_counts: dict[str, int]
    outcomes: list[CaseOutcome]

    def summary_lines(self) -> list[str]:
        lines = [
            f"Total cases:               {self.total_cases}",
            f"Generated successfully:    {self.generated}",
            f"Passed (all evaluators):   {self.passed}",
            f"Pass rate:                 {self.pass_rate:.1%}",
            f"Avg faithfulness score:    {self.avg_faithfulness:.3f}",
            f"Avg coverage score:        {self.avg_coverage:.3f}",
            f"Avg readability (grade):   {self.avg_readability_grade:.2f}",
            f"Avg latency per case:      {self.avg_latency_ms:.1f} ms",
            f"Unsupported claim rate:    {self.unsupported_claim_rate:.1%}",
        ]
        if self.avg_expected_factor_coverage is not None:
            lines.append(
                f"Expected-factor coverage:  {self.avg_expected_factor_coverage:.1%}"
            )
        if self.failure_reason_counts:
            reasons = ", ".join(
                f"{k}={v}" for k, v in sorted(self.failure_reason_counts.items())
            )
            lines.append(f"Failure reasons:           {reasons}")
        return lines


def _expected_coverage(
    case_id: str, covered: list[str], expected: dict | None
) -> float | None:
    if not expected or case_id not in expected:
        return None
    important = expected[case_id].get("expected_important_factors", [])
    if not important:
        return None
    covered_set = {c.lower() for c in covered}
    hits = sum(1 for name in important if name.lower() in covered_set)
    return hits / len(important)


def _evaluate_case(
    raw: dict, client: LLMClient, expected: dict | None
) -> CaseOutcome:
    case = RiskCase.model_validate(raw)
    start = time.perf_counter()
    try:
        explanation = generate_explanation(case, client)
        generated = True
    except LLMOutputValidationError:
        explanation = None
        generated = False
    latency_ms = (time.perf_counter() - start) * 1000

    if explanation is None:
        return CaseOutcome(
            case_id=case.case_id,
            passed=False,
            generated=False,
            faithfulness_score=0.0,
            coverage_score=0.0,
            readability_grade=0.0,
            safety_passed=False,
            unsupported_claim_count=0,
            expected_factor_coverage=None,
            failure_reasons=["generation_failed"],
            latency_ms=round(latency_ms, 2),
        )

    result: EvaluationResult = evaluate_explanation(explanation, case)
    return CaseOutcome(
        case_id=case.case_id,
        passed=result.passed,
        generated=generated,
        faithfulness_score=result.faithfulness.faithfulness_score,
        coverage_score=result.coverage.coverage_score,
        readability_grade=result.readability.grade_level,
        safety_passed=result.safety.passed,
        unsupported_claim_count=len(result.faithfulness.unsupported_claims),
        expected_factor_coverage=_expected_coverage(
            case.case_id, result.coverage.covered_factors, expected
        ),
        failure_reasons=result.failure_reasons(),
        latency_ms=round(latency_ms, 2),
    )


def run_batch(
    cases: list[dict],
    client: LLMClient | None = None,
    expected: dict | None = None,
) -> BatchReport:
    """Evaluate all cases and aggregate metrics. Pure given a deterministic client."""
    client = client or get_llm_client()
    outcomes = [_evaluate_case(raw, client, expected) for raw in cases]

    total = len(outcomes)
    generated = sum(1 for o in outcomes if o.generated)
    passed = sum(1 for o in outcomes if o.passed)
    reason_counts: Counter[str] = Counter()
    for o in outcomes:
        reason_counts.update(o.failure_reasons)

    expected_covs = [
        o.expected_factor_coverage
        for o in outcomes
        if o.expected_factor_coverage is not None
    ]

    def _avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    return BatchReport(
        total_cases=total,
        generated=generated,
        passed=passed,
        pass_rate=round(passed / total, 4) if total else 0.0,
        avg_faithfulness=_avg([o.faithfulness_score for o in outcomes]),
        avg_coverage=_avg([o.coverage_score for o in outcomes]),
        avg_readability_grade=_avg([o.readability_grade for o in outcomes]),
        avg_latency_ms=_avg([o.latency_ms for o in outcomes]),
        unsupported_claim_rate=(
            round(sum(1 for o in outcomes if o.unsupported_claim_count) / total, 4)
            if total
            else 0.0
        ),
        avg_expected_factor_coverage=_avg(expected_covs) if expected_covs else None,
        failure_reason_counts=dict(reason_counts),
        outcomes=outcomes,
    )


def _load_expected(input_path: Path) -> dict | None:
    expected_path = input_path.with_name("expected_outputs.json")
    if expected_path.exists():
        return json.loads(expected_path.read_text())
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ModelLens batch evaluation")
    parser.add_argument("--input", required=True, help="Path to risk_cases.json")
    parser.add_argument(
        "--output", default="eval_results.json", help="Where to write results"
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    cases = json.loads(input_path.read_text())
    expected = _load_expected(input_path)

    report = run_batch(cases, expected=expected)

    print("\n".join(report.summary_lines()))
    Path(args.output).write_text(report.model_dump_json(indent=2) + "\n")
    print(f"\nResults written to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
