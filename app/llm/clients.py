"""LLM provider clients.

A client is a thin text-in / text-out interface. The `fake` provider is the
default: it produces deterministic, grounded JSON with no API key or network,
so the whole pipeline and test suite run offline. Set `LLM_PROVIDER=openai` or
`anthropic` (with the matching key) for real generation.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from app.core.config import Settings, get_settings
from app.core.errors import ExplanationGenerationError
from app.schemas.explanation import ExplanationOutput, FactorExplanation

_JSON_BLOCK = re.compile(r"```json\s*(\{.*\})\s*```", re.DOTALL)


class LLMClient(Protocol):
    def complete(self, *, system: str, user: str) -> str:
        """Return the model's raw text response."""
        ...


def _extract_case_payload(user: str) -> dict:
    """Pull the embedded structured case JSON out of a user prompt."""
    match = _JSON_BLOCK.search(user)
    if not match:
        raise ExplanationGenerationError("no structured case found in prompt")
    return json.loads(match.group(1))


class FakeLLMClient:
    """Deterministic, grounded stub. Echoes evidence back so output is faithful."""

    def complete(self, *, system: str, user: str) -> str:
        case = _extract_case_payload(user)
        factors = case.get("factors", [])

        factor_explanations = [
            FactorExplanation(
                factor_name=f["name"],
                plain_english_reason=(
                    f"This contributed because {f['evidence']}"
                ),
                source_evidence_used=[f["evidence"]],
            )
            for f in factors
        ]
        band = case.get("risk_band", "unknown")
        drivers = ", ".join(f["name"] for f in factors) or "the available factors"
        output = ExplanationOutput(
            summary=(
                f"This case was placed in the {band} risk band, based mainly on "
                f"{drivers}."
            ),
            member_explanation=(
                f"Your risk was assessed as {band}. The main reasons were "
                f"{drivers}. " + " ".join(
                    f"For {f['name']}, {f['evidence']}" for f in factors
                )
            ).strip(),
            factor_explanations=factor_explanations,
            limitations=[
                "This explanation is based only on the information provided.",
                "It is not medical, legal, or financial advice.",
            ],
        )
        return output.model_dump_json()


class _OpenAIClient:
    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.llm_model

    def complete(self, *, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""


class _AnthropicClient:
    def __init__(self, settings: Settings) -> None:
        from anthropic import Anthropic

        self._client = Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.llm_model

    def complete(self, *, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=0.2,
        )
        return "".join(
            block.text for block in resp.content if block.type == "text"
        )


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """Build the configured LLM client."""
    settings = settings or get_settings()
    provider = settings.llm_provider.lower()
    if provider == "fake":
        return FakeLLMClient()
    if provider == "openai":
        return _OpenAIClient(settings)
    if provider == "anthropic":
        return _AnthropicClient(settings)
    raise ExplanationGenerationError(f"unknown LLM provider: {provider!r}")
