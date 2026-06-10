"""Read/response schemas for persisted risk cases.

Separate from the input `RiskCase` so the API can expose stored identifiers
(db id, member_id, timestamps) without polluting the validation contract.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.schemas.risk_case import FactorDirection, RiskBand


class RiskFactorRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    direction: FactorDirection
    weight: float
    evidence: str


class RiskCaseRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    case_id: str
    member_id: str
    risk_score: float
    risk_band: RiskBand
    model_name: str
    model_version: str
    created_at: datetime
    factors: list[RiskFactorRead]
