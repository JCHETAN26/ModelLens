"""Readability evaluation (build-plan §13, Task 6.3).

Flags explanations that are too technical for a member audience using
Flesch-Kincaid grade level, average sentence length, and a jargon word list.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.evals.text import flesch_kincaid_grade, split_sentences, tokenize
from app.schemas.eval import ReadabilityResult
from app.schemas.explanation import ExplanationOutput

# Domain/technical terms that don't belong in a member-facing explanation.
_JARGON = frozenset(
    {
        "coefficient",
        "gradient",
        "logit",
        "logistic",
        "regression",
        "stochastic",
        "hyperparameter",
        "covariate",
        "percentile",
        "quantile",
        "heuristic",
        "vector",
        "embedding",
        "propensity",
        "multivariate",
        "eigenvalue",
        "posterior",
        "bayesian",
    }
)

_MAX_AVG_SENTENCE_LEN = 28.0


def evaluate_readability(
    explanation: ExplanationOutput,
    max_grade: float | None = None,
) -> ReadabilityResult:
    max_grade = (
        max_grade if max_grade is not None else get_settings().readability_max_grade
    )

    text = f"{explanation.summary} {explanation.member_explanation}"
    grade = flesch_kincaid_grade(text)

    sentences = split_sentences(text)
    words = tokenize(text)
    avg_sentence_len = len(words) / len(sentences) if sentences else 0.0

    jargon = sorted({w for w in tokenize(text) if w in _JARGON})

    passed = (
        grade <= max_grade
        and avg_sentence_len <= _MAX_AVG_SENTENCE_LEN
        and not jargon
    )

    return ReadabilityResult(
        grade_level=grade,
        avg_sentence_length=round(avg_sentence_len, 2),
        jargon_terms=jargon,
        passed=passed,
    )
