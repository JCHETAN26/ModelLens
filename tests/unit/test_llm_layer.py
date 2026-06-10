"""Tests for the LLM layer: fake client, parsing, and retry policy."""

from __future__ import annotations

import json

import pytest

from app.core.errors import LLMOutputValidationError
from app.llm.clients import FakeLLMClient
from app.llm.prompts import build_user_prompt
from app.llm.structured_outputs import generate_explanation, parse_explanation
from app.schemas.explanation import ExplanationOutput
from app.schemas.risk_case import RiskCase

CASE = RiskCase.model_validate(
    {
        "member_id": "mem_1",
        "case_id": "case_1",
        "risk_score": 0.8,
        "risk_band": "high",
        "model_name": "m",
        "model_version": "v1",
        "factors": [
            {
                "name": "Recent emergency visit",
                "direction": "increases_risk",
                "weight": 0.6,
                "evidence": "2 ER visits in the last 60 days.",
            },
            {
                "name": "Medication refill gap",
                "direction": "increases_risk",
                "weight": 0.4,
                "evidence": "Refill gap of 21 days detected.",
            },
        ],
    }
)


def test_fake_client_returns_grounded_json() -> None:
    raw = FakeLLMClient().complete(system="", user=build_user_prompt(CASE))
    out = parse_explanation(raw)
    assert isinstance(out, ExplanationOutput)
    # Every factor is explained and cites its evidence verbatim.
    names = {fe.factor_name for fe in out.factor_explanations}
    assert names == {"Recent emergency visit", "Medication refill gap"}
    for fe in out.factor_explanations:
        assert fe.source_evidence_used  # non-empty
    assert "high" in out.summary


def test_generate_explanation_with_fake_client() -> None:
    out = generate_explanation(CASE, FakeLLMClient())
    assert len(out.factor_explanations) == 2
    assert out.limitations


def test_parse_rejects_non_json() -> None:
    with pytest.raises(LLMOutputValidationError):
        parse_explanation("this is not json")


def test_parse_rejects_schema_violation() -> None:
    # Missing required member_explanation.
    with pytest.raises(LLMOutputValidationError):
        parse_explanation(json.dumps({"summary": "x", "factor_explanations": []}))


def test_parse_tolerates_markdown_fences() -> None:
    valid = ExplanationOutput(
        summary="s", member_explanation="m", factor_explanations=[], limitations=[]
    ).model_dump_json()
    fenced = f"```json\n{valid}\n```"
    out = parse_explanation(fenced)
    assert out.summary == "s"


class _SequenceClient:
    """Returns canned responses in order, to exercise retry logic."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls = 0

    def complete(self, *, system: str, user: str) -> str:
        resp = self._responses[self.calls]
        self.calls += 1
        return resp


def test_retry_recovers_after_one_bad_response() -> None:
    good = ExplanationOutput(
        summary="s", member_explanation="m", factor_explanations=[], limitations=[]
    ).model_dump_json()
    client = _SequenceClient(["garbage, not json", good])
    out = generate_explanation(CASE, client)
    assert out.summary == "s"
    assert client.calls == 2  # one retry happened


def test_rejected_after_two_failures() -> None:
    client = _SequenceClient(["bad once", "bad twice"])
    with pytest.raises(LLMOutputValidationError):
        generate_explanation(CASE, client)
    assert client.calls == 2  # exactly two attempts, no third
