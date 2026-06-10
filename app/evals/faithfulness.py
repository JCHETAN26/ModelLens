"""Faithfulness evaluation (build-plan §13, Task 6.1).

Deterministic, lexical-overlap baseline: every claim in the explanation must be
traceable to the structured evidence. Detects two failure modes:
  1. Fabricated evidence citations (source_evidence_used not in the case).
  2. Claims whose content words don't overlap any case evidence/factor.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.evals.text import best_overlap, content_tokens
from app.schemas.eval import FaithfulnessResult
from app.schemas.explanation import ExplanationOutput
from app.schemas.risk_case import RiskCase

# Minimum content-word overlap for a claim to count as grounded.
_SUPPORT_OVERLAP = 0.3


def evaluate_faithfulness(
    explanation: ExplanationOutput,
    case: RiskCase,
    threshold: float | None = None,
) -> FaithfulnessResult:
    threshold = (
        threshold if threshold is not None else get_settings().faithfulness_threshold
    )

    case_evidence = [f.evidence for f in case.factors]
    evidence_set = {e.strip() for e in case_evidence}
    factor_name_tokens = content_tokens(" ".join(f.name for f in case.factors))

    # Grounding context = all structured input the explanation may legitimately
    # draw on: factor evidence, factor names, and the risk band.
    grounding = case_evidence + [f.name for f in case.factors] + [case.risk_band.value]

    supported: list[str] = []
    unsupported: list[str] = []

    # 1) Each factor explanation is a claim; its cited evidence must be real and
    #    its reasoning must overlap the structured input.
    for fe in explanation.factor_explanations:
        fabricated = [
            cited for cited in fe.source_evidence_used if cited.strip() not in evidence_set
        ]
        claim = fe.plain_english_reason
        if fabricated:
            unsupported.append(f"{fe.factor_name}: fabricated evidence citation")
            continue
        if best_overlap(claim, grounding) >= _SUPPORT_OVERLAP:
            supported.append(claim)
        else:
            unsupported.append(claim)

    # 2) Each sentence of the member explanation must overlap evidence OR
    #    reference a known factor name.
    from app.evals.text import split_sentences

    for sentence in split_sentences(explanation.member_explanation):
        toks = content_tokens(sentence)
        if not toks:
            continue
        mentions_factor = bool(toks & factor_name_tokens)
        grounded = best_overlap(sentence, grounding) >= _SUPPORT_OVERLAP
        if mentions_factor or grounded:
            supported.append(sentence)
        else:
            unsupported.append(sentence)

    total = len(supported) + len(unsupported)
    score = 1.0 if total == 0 else len(supported) / total
    passed = score >= threshold and not any(
        "fabricated" in u for u in unsupported
    )

    return FaithfulnessResult(
        faithfulness_score=score,
        supported_claims=supported,
        unsupported_claims=unsupported,
        passed=passed,
    )
