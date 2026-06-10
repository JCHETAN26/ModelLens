"""Coverage evaluation (build-plan §13, Task 6.2).

Weight-aware: a factor "appears" if it is explained or named in the member
explanation. The score is the fraction of total factor weight that is covered,
so missing a high-weight factor hurts more than missing a low-weight one.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.evals.text import content_tokens
from app.schemas.eval import CoverageResult
from app.schemas.explanation import ExplanationOutput
from app.schemas.risk_case import RiskCase


def _factor_is_covered(
    factor_name: str, explained_names: set[str], explanation_text: str
) -> bool:
    name_lower = factor_name.lower()
    if name_lower in explained_names:
        return True
    # Fallback: most content words of the factor name appear in the prose.
    name_tokens = content_tokens(factor_name)
    if not name_tokens:
        return False
    text_tokens = content_tokens(explanation_text)
    return name_tokens.issubset(text_tokens)


def evaluate_coverage(
    explanation: ExplanationOutput,
    case: RiskCase,
    threshold: float | None = None,
) -> CoverageResult:
    threshold = (
        threshold if threshold is not None else get_settings().coverage_threshold
    )

    explained_names = {
        fe.factor_name.strip().lower() for fe in explanation.factor_explanations
    }
    prose = f"{explanation.summary} {explanation.member_explanation}"

    total_weight = sum(f.weight for f in case.factors)
    covered_weight = 0.0
    covered: list[str] = []
    missing: list[str] = []

    for factor in case.factors:
        if _factor_is_covered(factor.name, explained_names, prose):
            covered.append(factor.name)
            covered_weight += factor.weight
        else:
            missing.append(factor.name)

    score = 1.0 if total_weight == 0 else covered_weight / total_weight
    return CoverageResult(
        coverage_score=score,
        covered_factors=covered,
        missing_factors=missing,
        passed=score >= threshold,
    )
