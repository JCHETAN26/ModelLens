# ModelLens — Resume-Safe Metrics

> **Rule:** this file records **only metrics actually measured by the implemented
> scripts.** No invented numbers. Each metric below is reproducible by running
> the batch evaluation.

## How these were measured

```bash
python -m app.evals.run_batch --input sample_data/risk_cases.json
```

- **Provider:** `fake` (deterministic, offline). These numbers therefore measure
  the **pipeline and evaluators**, not the quality of any specific hosted LLM.
- **Dataset:** 60 synthetic cases across 5 domains (healthcare readmission,
  insurance claim, credit, fraud, operational). Fully synthetic — no PHI/PII.
- **Date measured:** 2026-06-10.

## Measured results (fake provider, 60-case synthetic dataset)

| Metric | Value |
|--------|-------|
| Synthetic cases evaluated | 60 |
| Generated successfully | 60 / 60 (100%) |
| Passed all four evaluators | 50 / 60 (**83.3%**) |
| Average faithfulness score | **1.000** |
| Average coverage score | **1.000** |
| Average readability (FK grade) | **10.06** (range 6.9–13.8) |
| Unsupported-claim rate | **0.0%** |
| Expected-factor coverage | **100.0%** |
| Average latency per case | ~0.06 ms (fake provider; range 0.03–0.38 ms) |
| Failure reasons | readability = 10 (cases exceeding grade 12) |

The 10 readability failures are a genuine signal: the deterministic readability
evaluator flagged explanations whose Flesch-Kincaid grade exceeded the
member-facing threshold of 12 — demonstrating that the harness discriminates
rather than rubber-stamping.

## Upstream risk model (real data — German Credit)

A logistic-regression credit-risk model trained on the public German Credit
dataset (1,000 real applicants, no PII), whose per-feature contributions become
ModelLens risk cases.

```bash
python -m app.risk_model.generate --n 50 --output sample_data/real_risk_cases.json
python -m app.evals.run_batch --input sample_data/real_risk_cases.json
```

| Metric | Value |
|--------|-------|
| Dataset | German Credit, 1,000 applicants (750 train / 250 test) |
| Model | Logistic regression (exact coefficient × value attribution) |
| ROC-AUC (held-out) | **0.804** |
| Accuracy (held-out) | **0.772** |
| Real cases explained + evaluated | 50 |
| Pass rate through the eval pipeline | 100% (faithfulness/coverage 1.000, readability grade 8.69) |

This demonstrates the full path: **real data → trained model → structured risk
outputs → grounded, evaluated, audited explanations.**

## Test coverage

- 65 automated tests (unit + integration), green in CI.
- Schema validation, API routes, DB migrations + linkage, LLM retry policy,
  all four evaluators, the LangGraph workflow (pass / retry / fail), dataset
  reproducibility, and the batch CLI.

## Resume bullet (safe to use today)

> Built ModelLens, a LangGraph-based AI explanation pipeline that converts
> structured risk model outputs into member-facing explanations, with automated
> grounding, coverage, readability, and safety evaluation. Trained a logistic-
> regression credit-risk model (ROC-AUC 0.80) on the German Credit dataset and
> fed its per-feature contributions through the pipeline, demonstrating an
> end-to-end path from raw data to grounded, audited explanations.

## Claims NOT to make (until separately measured)

The following are **not** supported by any implemented measurement and must not
appear on a resume unless/until real benchmarks produce them:

- ❌ "65% analyst review-time reduction" — no human-time benchmark exists.
- ❌ "85% grounding accuracy" — faithfulness here is a lexical baseline measured
  against the `fake` provider, not a labeled accuracy study.
- ❌ "200-case evaluation set" — the dataset is 60 synthetic cases.
- ❌ Any latency claim implying real-LLM performance — measured latency reflects
  the deterministic `fake` provider.

When real LLM providers and labeled data are introduced, re-run the batch eval
and update this table with the new measured figures.
