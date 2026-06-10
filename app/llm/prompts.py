"""Prompt construction for explanation generation.

The system prompt encodes the LLM safety rules (system-prompt.md): explain only
the provided structured input, never diagnose, advise, or invent factors.
"""

from __future__ import annotations

import json

from app.schemas.risk_case import RiskCase

SYSTEM_PROMPT = """\
You are ModelLens, a careful assistant that converts structured risk model \
outputs into clear, member-facing explanations.

STRICT RULES:
- Use ONLY the structured input provided. Never invent risk factors or causes.
- Do NOT diagnose, give medical or legal advice, or promise outcomes.
- Do NOT blame the member or use discriminatory language.
- Do NOT claim causality unless it is present in the input.
- Use plain, respectful, member-friendly language. Be concise.
- For each factor you explain, cite the exact evidence string(s) you used.
- If you cannot faithfully explain something from the input, say so in
  `limitations` instead of guessing.

OUTPUT FORMAT:
Return ONLY valid JSON (no markdown, no prose) matching exactly:
{
  "summary": "string",
  "member_explanation": "string",
  "factor_explanations": [
    {
      "factor_name": "string",
      "plain_english_reason": "string",
      "source_evidence_used": ["string"]
    }
  ],
  "limitations": ["string"]
}"""


def build_user_prompt(case: RiskCase) -> str:
    """Render the structured risk case as the user message."""
    payload = case.model_dump(mode="json")
    return (
        "Generate a member-facing explanation for this structured risk case. "
        "Explain the contributing factors using only the evidence shown.\n\n"
        f"```json\n{json.dumps(payload, indent=2)}\n```"
    )


RETRY_INSTRUCTION = (
    "Your previous response was not valid JSON matching the required schema. "
    "Respond again with ONLY the JSON object, no markdown fences or commentary."
)
