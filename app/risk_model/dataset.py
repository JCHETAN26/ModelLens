"""German Credit dataset loader + human-readable feature metadata.

Real, public, no-PII dataset (UCI Statlog / OpenML `credit-g`): 1,000 loan
applicants labeled good/bad credit risk. The CSV is committed under `data/` so
training is fully offline and reproducible; if it is ever missing we fall back
to fetching it from OpenML and caching it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "german_credit.csv"

TARGET = "class"
RISK_LABEL = "bad"  # the positive (higher-risk) class

NUMERIC_FEATURES = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]

CATEGORICAL_FEATURES = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
]

# Friendly names for evidence/factor strings shown to a member.
READABLE_NAMES: dict[str, str] = {
    "checking_status": "Checking account status",
    "duration": "Loan duration (months)",
    "credit_history": "Credit history",
    "purpose": "Loan purpose",
    "credit_amount": "Requested credit amount",
    "savings_status": "Savings account status",
    "employment": "Employment length",
    "installment_commitment": "Installment rate (% of income)",
    "personal_status": "Personal status",
    "other_parties": "Other debtors / guarantors",
    "residence_since": "Years at current residence",
    "property_magnitude": "Property owned",
    "age": "Age (years)",
    "other_payment_plans": "Other payment plans",
    "housing": "Housing situation",
    "existing_credits": "Existing credits at this bank",
    "job": "Job category",
    "num_dependents": "Number of dependents",
    "own_telephone": "Has registered telephone",
    "foreign_worker": "Foreign worker",
}


def readable(feature: str) -> str:
    return READABLE_NAMES.get(feature, feature.replace("_", " ").capitalize())


def load_german_credit() -> pd.DataFrame:
    """Load the committed dataset, fetching + caching from OpenML if absent."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)

    from sklearn.datasets import fetch_openml  # local import: only needed on miss

    frame = fetch_openml(
        "credit-g", version=1, as_frame=True, parser="pandas"
    ).frame
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_PATH, index=False)
    return frame
