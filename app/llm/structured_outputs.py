"""Parse and validate LLM output into the strict explanation schema.

Generation policy (build-plan §11):
- The model must return valid JSON matching `ExplanationOutput`.
- Invalid output triggers exactly one retry with a corrective instruction.
- If it fails validation twice, generation is rejected (raises).
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from app.core.errors import LLMOutputValidationError
from app.core.logging import get_logger
from app.llm.clients import LLMClient
from app.llm.prompts import RETRY_INSTRUCTION, SYSTEM_PROMPT, build_user_prompt
from app.schemas.explanation import ExplanationOutput
from app.schemas.risk_case import RiskCase

logger = get_logger(__name__)


def _strip_fences(raw: str) -> str:
    """Tolerate ```json fenced output some models emit."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text
        text = text.removeprefix("json").strip()
        if text.endswith("```"):
            text = text[: -3].strip()
    return text


def parse_explanation(raw: str) -> ExplanationOutput:
    """Parse raw model text into a validated ExplanationOutput. Raises on failure."""
    text = _strip_fences(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMOutputValidationError(f"output is not valid JSON: {exc}") from exc
    try:
        return ExplanationOutput.model_validate(data)
    except ValidationError as exc:
        raise LLMOutputValidationError(f"output failed schema validation: {exc}") from exc


def generate_explanation(case: RiskCase, client: LLMClient) -> ExplanationOutput:
    """Generate a validated explanation, retrying once on invalid output.

    Raises LLMOutputValidationError if the model fails validation twice.
    """
    system = SYSTEM_PROMPT
    user = build_user_prompt(case)

    attempts = 2  # initial attempt + one retry
    last_error: LLMOutputValidationError | None = None

    for attempt in range(1, attempts + 1):
        raw = client.complete(system=system, user=user)
        try:
            return parse_explanation(raw)
        except LLMOutputValidationError as exc:
            last_error = exc
            logger.warning("LLM output invalid (attempt %d/%d): %s", attempt, attempts, exc)
            # Add a corrective instruction for the retry.
            user = f"{build_user_prompt(case)}\n\n{RETRY_INSTRUCTION}"

    raise LLMOutputValidationError(
        f"explanation rejected after {attempts} attempts: {last_error}"
    )
