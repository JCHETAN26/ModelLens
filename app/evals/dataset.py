"""Synthetic risk-case dataset generator (build-plan §14).

Deterministic (seeded) generation of fully synthetic risk cases across five
domains. NO real patient/member/customer data, no PHI/PII — every value here is
fabricated from templates.

Run as a module to (re)write the dataset:

    python -m app.evals.dataset
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from app.schemas.risk_case import RiskBand, RiskCase

DATA_DIR = Path(__file__).resolve().parents[2] / "sample_data"
CASES_PATH = DATA_DIR / "risk_cases.json"
EXPECTED_PATH = DATA_DIR / "expected_outputs.json"

# Candidate factors per domain: (name, direction, evidence, base_weight).
_FACTOR_POOLS: dict[str, tuple[str, list[tuple[str, str, str, float]]]] = {
    "healthcare_readmission": (
        "readmission_risk",
        [
            ("Recent emergency visit", "increases_risk",
             "Member had two emergency visits in the last sixty days.", 0.30),
            ("Medication refill gap", "increases_risk",
             "A refill gap of twenty one days was detected for a prescribed medication.", 0.22),
            ("Multiple chronic conditions", "increases_risk",
             "Three long term conditions are recorded in the care summary.", 0.25),
            ("Missed follow up appointment", "increases_risk",
             "One scheduled follow up appointment was not attended last month.", 0.18),
            ("Stable vitals at last visit", "decreases_risk",
             "Vital signs were within the expected range at the most recent visit.", 0.15),
            ("Enrolled in care program", "decreases_risk",
             "Member is actively enrolled in a care management program.", 0.12),
        ],
    ),
    "insurance_claim": (
        "claim_risk",
        [
            ("High claim amount", "increases_risk",
             "The submitted claim amount is well above the average for this policy type.", 0.28),
            ("Recent policy change", "increases_risk",
             "The policy coverage was changed within the last thirty days before the claim.", 0.20),
            ("Multiple claims this year", "increases_risk",
             "Four separate claims have been filed in the current policy year.", 0.24),
            ("Incomplete documentation", "increases_risk",
             "Required supporting documents were missing from the initial submission.", 0.16),
            ("Long tenure with insurer", "decreases_risk",
             "The policy has been active and in good standing for eight years.", 0.14),
            ("Verified provider", "decreases_risk",
             "The service provider is on the verified in network list.", 0.10),
        ],
    ),
    "credit_risk": (
        "credit_risk",
        [
            ("High credit utilization", "increases_risk",
             "Reported revolving credit utilization is above eighty percent.", 0.30),
            ("Recent missed payment", "increases_risk",
             "One payment was reported as thirty days late in the last quarter.", 0.26),
            ("Short credit history", "increases_risk",
             "The credit file has been open for less than two years.", 0.18),
            ("Multiple recent inquiries", "increases_risk",
             "Five hard inquiries were recorded in the last six months.", 0.16),
            ("Stable income on file", "decreases_risk",
             "A consistent monthly income has been verified for two years.", 0.15),
            ("Low total balance", "decreases_risk",
             "The total outstanding balance is small relative to available credit.", 0.12),
        ],
    ),
    "fraud_risk": (
        "fraud_risk",
        [
            ("Unusual transaction location", "increases_risk",
             "A transaction occurred in a country not previously seen on the account.", 0.32),
            ("Rapid successive transactions", "increases_risk",
             "Six transactions were attempted within two minutes.", 0.27),
            ("New device login", "increases_risk",
             "The session originated from a device not previously linked to the account.", 0.20),
            ("Mismatched billing details", "increases_risk",
             "The billing postal code did not match the records on file.", 0.18),
            ("Long standing account", "decreases_risk",
             "The account has a five year history with no prior disputes.", 0.13),
            ("Transaction within usual range", "decreases_risk",
             "The transaction amount is consistent with the typical spending pattern.", 0.11),
        ],
    ),
    "operational_risk": (
        "operational_risk",
        [
            ("Repeated process exceptions", "increases_risk",
             "The workflow recorded several manual exception overrides this week.", 0.29),
            ("Aging open tickets", "increases_risk",
             "Multiple support tickets have remained open past the target resolution time.", 0.23),
            ("Single point of dependency", "increases_risk",
             "A critical step depends on one unbacked component.", 0.21),
            ("Recent configuration change", "increases_risk",
             "A configuration change was deployed shortly before the incident window.", 0.17),
            ("Documented runbook exists", "decreases_risk",
             "A current runbook is available for the affected process.", 0.14),
            ("Redundant capacity available", "decreases_risk",
             "Spare capacity is provisioned and verified for failover.", 0.12),
        ],
    ),
}


def _band(score: float) -> RiskBand:
    if score >= 0.66:
        return RiskBand.HIGH
    if score >= 0.33:
        return RiskBand.MEDIUM
    return RiskBand.LOW


def _build_case(category: str, idx: int, rng: random.Random) -> RiskCase:
    model_name, pool = _FACTOR_POOLS[category]
    n_factors = rng.randint(2, 4)
    chosen = rng.sample(pool, n_factors)

    factors = []
    score_signal = 0.0
    for name, direction, evidence, base in chosen:
        weight = round(min(0.95, max(0.05, base + rng.uniform(-0.05, 0.05))), 2)
        factors.append(
            {
                "name": name,
                "direction": direction,
                "weight": weight,
                "evidence": evidence,
            }
        )
        score_signal += weight if direction == "increases_risk" else -weight

    # Map the net signal into a [0, 1] risk score.
    risk_score = round(min(0.98, max(0.02, 0.5 + score_signal * 0.6)), 2)

    return RiskCase.model_validate(
        {
            "member_id": f"mem_{category[:3]}_{idx:03d}",
            "case_id": f"{category}_{idx:03d}",
            "risk_score": risk_score,
            "risk_band": _band(risk_score).value,
            "model_name": f"{model_name}_v1",
            "model_version": "2026.01",
            "factors": factors,
        }
    )


def build_dataset(
    per_category: int = 12, seed: int = 7
) -> tuple[list[dict], dict[str, dict]]:
    """Return (cases, expected_outputs). Deterministic for a given seed."""
    rng = random.Random(seed)
    cases: list[dict] = []
    expected: dict[str, dict] = {}

    for category in _FACTOR_POOLS:
        for idx in range(1, per_category + 1):
            case = _build_case(category, idx, rng)
            cases.append(case.model_dump(mode="json"))
            # Expected important factors = top half by weight (at least one).
            top = case.top_factors()
            keep = max(1, len(top) // 2)
            expected[case.case_id] = {
                "category": category,
                "expected_important_factors": [f.name for f in top[:keep]],
            }
    return cases, expected


def write_dataset(per_category: int = 12, seed: int = 7) -> int:
    cases, expected = build_dataset(per_category, seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CASES_PATH.write_text(json.dumps(cases, indent=2) + "\n")
    EXPECTED_PATH.write_text(json.dumps(expected, indent=2) + "\n")
    return len(cases)


if __name__ == "__main__":  # pragma: no cover
    count = write_dataset()
    print(f"Wrote {count} synthetic cases to {CASES_PATH}")
    print(f"Wrote expectations to {EXPECTED_PATH}")
