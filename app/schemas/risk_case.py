"""Structured risk case input schema.

This is the contract for data entering ModelLens. Validation here is the first
line of defense: a malformed or under-specified case must never reach the LLM.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FactorDirection(StrEnum):
    """Whether a factor pushes the risk score up or down."""

    INCREASES_RISK = "increases_risk"
    DECREASES_RISK = "decreases_risk"


class RiskBand(StrEnum):
    """Coarse categorical risk level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskFactor(BaseModel):
    """A single contributing factor with its supporting evidence.

    `evidence` is required: it is the grounding source the generated explanation
    must trace back to. A factor without evidence cannot be faithfully explained.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Human-readable factor name.")
    direction: FactorDirection
    weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Relative contribution of this factor in [0, 1].",
    )
    evidence: str = Field(
        min_length=1,
        description="Structured evidence snippet supporting this factor.",
    )

    @field_validator("name", "evidence")
    @classmethod
    def _strip_and_require(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


class RiskCase(BaseModel):
    """A complete structured risk model output for one member/case."""

    model_config = ConfigDict(extra="forbid")

    member_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    risk_score: float = Field(
        ge=0.0, le=1.0, description="Model risk score in [0, 1]."
    )
    risk_band: RiskBand
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    factors: list[RiskFactor] = Field(
        min_length=1, description="At least one contributing factor is required."
    )

    @field_validator("factors")
    @classmethod
    def _unique_factor_names(cls, factors: list[RiskFactor]) -> list[RiskFactor]:
        names = [f.name.lower() for f in factors]
        if len(names) != len(set(names)):
            raise ValueError("factor names must be unique within a case")
        return factors

    def top_factors(self, k: int | None = None) -> list[RiskFactor]:
        """Return factors sorted by descending weight (highest impact first)."""
        ordered = sorted(self.factors, key=lambda f: f.weight, reverse=True)
        return ordered if k is None else ordered[:k]
