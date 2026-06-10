# ModelLens — Build Plan

> Production-style AI engineering project that converts structured risk model outputs into member-facing, plain-English explanations with automated grounding, safety, and faithfulness evaluation.

---

## 1. Product Goal

ModelLens takes structured risk model outputs such as risk scores, contributing factors, evidence snippets, and model metadata, then generates clear, faithful, member-friendly explanations.

The system must not hallucinate. Every generated explanation must be traceable to structured input data.

---

## 2. Core User Story

As a healthcare, insurance, or fintech analyst, I want to convert structured risk model outputs into compliant, understandable explanations so that members or internal teams can understand why a risk score was assigned without requiring manual analyst writing.

---

## 3. Architecture Diagram

```text
┌──────────────────────────┐
│   Structured Risk Input  │
│  JSON / CSV / API Upload │
└─────────────┬────────────┘
              │
              v
┌──────────────────────────┐
│ FastAPI Backend          │
│ - Validate payload       │
│ - Normalize factors      │
│ - Store request          │
└─────────────┬────────────┘
              │
              v
┌──────────────────────────┐
│ PostgreSQL               │
│ - members                │
│ - risk_cases             │
│ - explanations           │
│ - eval_results           │
│ - audit_logs             │
└─────────────┬────────────┘
              │
              v
┌──────────────────────────┐
│ LangGraph Workflow       │
│ 1. Plan explanation      │
│ 2. Generate explanation  │
│ 3. Grounding check       │
│ 4. Safety check          │
│ 5. Rewrite if needed     │
│ 6. Persist result        │
└─────────────┬────────────┘
              │
              v
┌──────────────────────────┐
│ LLM Layer                │
│ OpenAI API / Claude      │
│ Structured JSON output   │
└─────────────┬────────────┘
              │
              v
┌──────────────────────────┐
│ Evaluation Harness       │
│ - Faithfulness score     │
│ - Unsupported claims     │
│ - Coverage score         │
│ - Readability score      │
│ - Safety flags           │
└─────────────┬────────────┘
              │
              v
┌──────────────────────────┐
│ Analyst/API Output       │
│ - Explanation            │
│ - Evidence mapping       │
│ - Eval score             │
│ - Audit trail            │
└──────────────────────────┘
```

---

## 4. Tech Stack

* Python 3.11+
* FastAPI
* LangGraph
* OpenAI API
* Optional Claude API
* PostgreSQL
* SQLAlchemy or SQLModel
* Pydantic
* Alembic
* Pytest
* Ruff
* Docker Compose
* GitHub Actions
* Claude Code for implementation assistance

---

## 5. Repository Structure

```text
ModelLens/
├── app/
│   ├── api/
│   │   ├── routes_health.py
│   │   ├── routes_cases.py
│   │   └── routes_explanations.py
│   ├── core/
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── errors.py
│   ├── db/
│   │   ├── session.py
│   │   ├── models.py
│   │   └── migrations/
│   ├── schemas/
│   │   ├── risk_case.py
│   │   ├── explanation.py
│   │   └── eval.py
│   ├── llm/
│   │   ├── clients.py
│   │   ├── prompts.py
│   │   └── structured_outputs.py
│   ├── graph/
│   │   ├── state.py
│   │   ├── nodes.py
│   │   └── workflow.py
│   ├── evals/
│   │   ├── faithfulness.py
│   │   ├── coverage.py
│   │   ├── readability.py
│   │   └── safety.py
│   └── main.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── sample_data/
│   ├── risk_cases.json
│   └── expected_outputs.json
├── docs/
│   ├── architecture.md
│   ├── eval_methodology.md
│   └── resume_metrics.md
├── .github/
│   └── workflows/
│       └── ci.yml
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── README.md
├── build-plan.md
└── system-prompt.md
```

---

## 6. Git Workflow Rules

No direct pushes to `main`.

Every change must follow this workflow:

```text
main
 └── feature/<task-name>
        └── pull request
              └── CI passes
                    └── review
                          └── squash merge into main
```

Required branch naming:

```text
feature/init-project
feature/db-schema
feature/risk-case-api
feature/langgraph-workflow
feature/eval-harness
feature/docker-ci
```

Required PR checklist:

```text
- [ ] Code is isolated to one task
- [ ] Tests added or updated
- [ ] Ruff passes
- [ ] Pytest passes
- [ ] No secrets committed
- [ ] README/docs updated if behavior changed
- [ ] No direct push to main
```

---

## 7. Phase 0 — Project Initialization

### Task 0.1 — Initialize Repo

Create:

* Python project
* `pyproject.toml`
* FastAPI app skeleton
* `.env.example`
* `.gitignore`
* `README.md`
* `build-plan.md`
* `system-prompt.md`

Acceptance:

```text
- App starts locally with uvicorn
- `/health` returns 200
- No secrets committed
- Project installs cleanly
```

---

## 8. Phase 1 — Data Model

### Task 1.1 — Define Input Schema

Create a structured risk case schema.

Example input:

```json
{
  "member_id": "mem_123",
  "case_id": "case_001",
  "risk_score": 0.87,
  "risk_band": "high",
  "model_name": "readmission_risk_v1",
  "model_version": "2026.01",
  "factors": [
    {
      "name": "Recent emergency visit",
      "direction": "increases_risk",
      "weight": 0.31,
      "evidence": "Member had 2 ER visits in the last 60 days."
    },
    {
      "name": "Medication refill gap",
      "direction": "increases_risk",
      "weight": 0.24,
      "evidence": "Refill gap of 21 days detected for prescribed medication."
    }
  ]
}
```

Acceptance:

```text
- Pydantic validation rejects missing required fields
- Risk score must be between 0 and 1
- Factors must include name, direction, and evidence
- Unit tests cover valid and invalid payloads
```

---

## 9. Phase 2 — Database Layer

### Task 2.1 — PostgreSQL Schema

Tables:

```text
members
risk_cases
risk_factors
explanations
eval_results
audit_logs
```

Acceptance:

```text
- Alembic migrations run successfully
- Risk case can be inserted and queried
- Explanation records link back to original case
- Eval records link back to explanation
```

---

## 10. Phase 3 — API Layer

### Task 3.1 — Risk Case Endpoints

Create:

```text
POST /cases
GET /cases/{case_id}
GET /cases
```

Acceptance:

```text
- Can create a risk case
- Can retrieve a risk case
- Invalid payloads return 422
- Tests cover all routes
```

### Task 3.2 — Explanation Endpoint

Create:

```text
POST /cases/{case_id}/explanations
GET /cases/{case_id}/explanations/latest
```

Acceptance:

```text
- Explanation generation can be triggered
- Response includes explanation text, grounding score, and status
- Failed generations return clear error states
```

---

## 11. Phase 4 — LLM Layer

### Task 4.1 — Structured Prompt Contract

LLM output must be strict JSON:

```json
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
}
```

Rules:

```text
- Do not mention unsupported reasons
- Do not diagnose
- Do not provide medical advice
- Do not claim causality unless present in input
- Use member-friendly language
- Keep explanation concise
```

Acceptance:

```text
- LLM client returns validated JSON
- Invalid JSON triggers one retry
- Output is rejected if schema validation fails twice
```

---

## 12. Phase 5 — LangGraph Workflow

### Task 5.1 — Build Graph State

State should include:

```text
case_id
risk_case
draft_explanation
eval_result
retry_count
final_explanation
status
errors
```

### Task 5.2 — Build Nodes

Nodes:

```text
load_case
generate_explanation
evaluate_faithfulness
evaluate_coverage
evaluate_safety
rewrite_explanation
persist_result
fail_gracefully
```

Graph flow:

```text
load_case
  -> generate_explanation
  -> evaluate_faithfulness
  -> evaluate_coverage
  -> evaluate_safety
  -> if pass: persist_result
  -> if fail and retry_count < 2: rewrite_explanation
  -> if fail and retry_count >= 2: fail_gracefully
```

Acceptance:

```text
- Workflow completes for valid case
- Workflow retries when grounding fails
- Workflow stops after max retries
- Every run produces an audit log
```

---

## 13. Phase 6 — Evaluation Harness

### Task 6.1 — Faithfulness Evaluation

Measure whether generated claims are supported by input factors.

Output:

```json
{
  "faithfulness_score": 0.91,
  "unsupported_claims": [],
  "supported_claims": ["..."],
  "decision": "pass"
}
```

Initial approach:

```text
- Extract claims from explanation
- Compare each claim against evidence strings
- Use deterministic lexical overlap first
- Add LLM-as-judge only after baseline works
```

Acceptance:

```text
- Unsupported claims are detected
- Faithfulness score is computed
- Threshold is configurable
- Default pass threshold: 0.85
```

### Task 6.2 — Coverage Evaluation

Measure whether all important input factors were explained.

Acceptance:

```text
- Top-weighted factors must appear in output
- Missing high-weight factors lower coverage score
```

### Task 6.3 — Readability Evaluation

Measure:

```text
- Grade level
- Sentence length
- Jargon
- Member-facing tone
```

Acceptance:

```text
- Output includes readability score
- Overly technical explanations are flagged
```

### Task 6.4 — Safety Evaluation

Reject:

```text
- Medical advice
- Legal advice
- Unsupported diagnoses
- Discriminatory language
- Absolute guarantees
- Blame-oriented wording
```

Acceptance:

```text
- Unsafe outputs fail
- Safety result is persisted
```

---

## 14. Phase 7 — Test Dataset

Create `sample_data/risk_cases.json` with at least 50 synthetic cases.

Case categories:

```text
- Healthcare readmission risk
- Insurance claim risk
- Credit risk
- Fraud risk
- Operational risk
```

Important:

```text
Use synthetic data only.
Do not use real patient/member/customer data.
Do not include PHI, PII, SSNs, addresses, or real medical records.
```

Acceptance:

```text
- 50+ synthetic cases exist
- Each case has expected important factors
- Dataset can run through batch eval script
```

---

## 15. Phase 8 — Batch Evaluation CLI

Create:

```bash
python -m app.evals.run_batch --input sample_data/risk_cases.json
```

Output:

```text
Total cases
Pass rate
Average faithfulness score
Average coverage score
Average readability score
Failure reasons
```

Acceptance:

```text
- Batch eval runs locally
- Results saved to `eval_results.json`
- Metrics are reproducible
```

---

## 16. Phase 9 — Docker + Local Dev

Create:

```text
Dockerfile
docker-compose.yml
```

Services:

```text
api
postgres
```

Acceptance:

```text
- `docker compose up` starts the app
- Migrations run successfully
- API can connect to database
```

---

## 17. Phase 10 — CI/CD

GitHub Actions must run on every PR:

```text
ruff check
pytest
schema validation tests
docker build
```

Acceptance:

```text
- CI blocks broken PRs
- Branch protection enabled on main
- PR required before merge
```

---

## 18. Phase 11 — Documentation

Create:

```text
README.md
docs/architecture.md
docs/eval_methodology.md
docs/resume_metrics.md
```

`resume_metrics.md` must track only real measured metrics.

Example:

```text
Do not claim:
- 65% review-time reduction
- 200-case eval set
- 85% grounding accuracy

Until measured by actual scripts.
```

Acceptance:

```text
- README explains setup
- Eval methodology explains scoring
- Resume metrics file records only verified results
```

---

## 19. Final Resume-Safe Metrics

Only after implementation, measure:

```text
- Number of synthetic cases evaluated
- Average faithfulness score
- Percentage of outputs passing threshold
- Average latency per explanation
- Retry rate
- Unsupported claim rate
- Manual review time estimate, if actually benchmarked
```

Resume bullet should eventually become:

```text
Built ModelLens, a LangGraph-based AI explanation pipeline that converts structured risk model outputs into member-facing explanations, with automated grounding, coverage, readability, and safety evaluation across synthetic risk cases.
```

Only add numeric claims after real measurement.

---

## 20. Definition of Done

ModelLens is complete when:

```text
- API accepts structured risk cases
- LangGraph generates explanations
- Evaluation harness scores every output
- Unsupported claims are detected
- Results are stored in PostgreSQL
- Batch eval produces reproducible metrics
- Docker setup works locally
- CI passes on every PR
- No direct pushes to main
- README explains the full system clearly
```
