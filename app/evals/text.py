"""Shared deterministic text utilities for evaluators.

Pure functions, no LLM calls — so eval results are reproducible.
"""

from __future__ import annotations

import re

_STOPWORDS = frozenset(
    """
    a an and are as at be been being but by for from had has have he her his in
    into is it its of on or that the their them they this to was were will with
    you your we our us i me my mr mrs because contributed based mainly main
    reasons reason was assessed risk
    """.split()
)

_WORD_RE = re.compile(r"[a-z0-9']+")
_SENTENCE_RE = re.compile(r"[.!?]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens."""
    return _WORD_RE.findall(text.lower())


def content_tokens(text: str) -> set[str]:
    """Content words (stopwords removed), as a set."""
    return {t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 1}


def overlap_ratio(claim: str, evidence: str) -> float:
    """Fraction of the claim's content words that appear in the evidence.

    Returns 1.0 for an empty claim (nothing unsupported to find).
    """
    claim_tokens = content_tokens(claim)
    if not claim_tokens:
        return 1.0
    evidence_tokens = content_tokens(evidence)
    if not evidence_tokens:
        return 0.0
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


def best_overlap(claim: str, evidences: list[str]) -> float:
    """Best overlap ratio of the claim against any evidence string."""
    if not evidences:
        return 0.0
    return max(overlap_ratio(claim, e) for e in evidences)


def split_sentences(text: str) -> list[str]:
    """Split text into non-empty, stripped sentences."""
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def count_syllables(word: str) -> int:
    """Heuristic syllable count for a single word (>=1)."""
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def flesch_kincaid_grade(text: str) -> float:
    """Flesch-Kincaid grade level. 0.0 for empty/degenerate text."""
    sentences = split_sentences(text)
    words = tokenize(text)
    if not sentences or not words:
        return 0.0
    syllables = sum(count_syllables(w) for w in words)
    wps = len(words) / len(sentences)
    spw = syllables / len(words)
    return round(0.39 * wps + 11.8 * spw - 15.59, 2)
