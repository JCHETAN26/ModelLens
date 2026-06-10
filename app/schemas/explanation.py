"""LLM explanation output schema — the strict JSON contract (build-plan §11).

The model must return exactly this shape. Anything else is rejected and retried.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FactorExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factor_name: str = Field(min_length=1)
    plain_english_reason: str = Field(min_length=1)
    source_evidence_used: list[str] = Field(default_factory=list)


class ExplanationOutput(BaseModel):
    """Structured, member-facing explanation produced by the LLM."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    member_explanation: str = Field(min_length=1)
    factor_explanations: list[FactorExplanation] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
