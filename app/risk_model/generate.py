"""Generate real RiskCases from the trained credit-risk model.

    python -m app.risk_model.generate --n 50 --output sample_data/real_risk_cases.json

The output is drop-in compatible with the explanation pipeline and the batch
evaluator, so you can run:

    python -m app.evals.run_batch --input sample_data/real_risk_cases.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.risk_model.adapter import case_from_row
from app.risk_model.train import train_risk_model


def generate_cases(n: int = 50, seed: int = 42) -> tuple[list[dict], dict]:
    """Train the model and build `n` RiskCases from held-out applicants."""
    model = train_risk_model(seed=seed)
    n = min(n, len(model.test_frame))

    cases: list[dict] = []
    for i in range(n):
        row = model.test_frame.iloc[i]
        case = case_from_row(
            model,
            row,
            case_id=f"credit_g_{i:04d}",
            member_id=f"applicant_{i:04d}",
        )
        cases.append(case.model_dump(mode="json"))

    return cases, model.metrics()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate RiskCases from the trained credit-risk model"
    )
    parser.add_argument("--n", type=int, default=50, help="Number of cases")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="sample_data/real_risk_cases.json",
        help="Where to write the generated cases",
    )
    args = parser.parse_args(argv)

    cases, metrics = generate_cases(n=args.n, seed=args.seed)

    print("Trained credit-risk model (German Credit, logistic regression):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cases, indent=2) + "\n")
    print(f"\nWrote {len(cases)} real risk cases to {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
