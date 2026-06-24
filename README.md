# ModelLens

> **Convert structured risk model outputs into faithful, member-facing plain-English explanations — with automated grounding, safety, and faithfulness evaluation built into the pipeline.**

ModelLens is a production-shaped AI engineering project. It takes the structured output of a risk model — a risk score, the factors that drove it, the evidence behind each factor, and model metadata — and turns it into a clear, member-friendly explanation. Critically, it is engineered **not to hallucinate**: every sentence of every explanation is traceable to the structured input, and each output is automatically scored for faithfulness, coverage, readability, and safety *before* it is ever stored or returned.

---

## Table of contents

- [The problem](#the-problem)
- [Why we built it this way](#why-we-built-it-this-way)
- [What ModelLens does](#what-modellens-does)
- [How it works (in detail)](#how-it-works-in-detail)
- [The evaluation harness](#the-evaluation-harness)
- [Findings & measured metrics](#findings--measured-metrics)
- [Real upstream risk model (end-to-end)](#real-upstream-risk-model-end-to-end)
- [Tech stack](#tech-stack)
- [Architecture & data model](#architecture--data-model)
- [Getting started](#getting-started)
- [API reference](#api-reference)
- [Project layout](#project-layout)
- [Engineering practices](#engineering-practices)
- [Safety & data policy](#safety--data-policy)
- [Limitations & honest scope](#limitations--honest-scope)

---

## The problem

Risk models in **healthcare, insurance, and fintech** produce outputs that are accurate but opaque: a score like `0.87`, a risk band like `high`, and a list of weighted feature contributions. Turning that into something a member — or an internal analyst — can actually understand is normally **manual writing work**: slow, inconsistent between analysts, and difficult to audit.

The obvious shortcut, *"just ask an LLM to explain the score,"* fails the bar these regulated domains require. A free-form LLM will:

- **hallucinate reasons** that aren't in the model's actual inputs,
- drift into **medical or legal advice**,
- imply **causality** the model never claimed, or
- use **technical jargon** a member can't parse.

In a compliance-sensitive setting, any one of those makes the output unusable. ModelLens exists to make LLM-generated risk explanations **trustworthy enough to ship**.

## Why we built it this way

Three principles shape every design decision:

1. **Grounded by construction.** The LLM is only permitted to explain factors and evidence that are present in the structured input. It is never asked to reason from scratch about the member.
2. **Verified, not assumed.** Every generated explanation is run through four **deterministic evaluators** before it can be marked `passed`. An output that fails is automatically rewritten and re-checked, or rejected — it is never silently returned. Evaluation is a first-class part of the system, not a bolt-on.
3. **Auditable.** Every run writes an audit log, and the data model links each case → explanation → evaluation result, so any output can always be traced back to the exact inputs that produced it.

A secondary goal was to build this as a **realistic reference project** rather than a notebook demo: typed schemas, real database migrations, a CI gate on every PR, Docker for local parity, branch protection on `main`, and a complete path from a *real trained model* all the way to a graded, audited explanation.

## What ModelLens does

```text
   Structured risk case            LangGraph workflow                Evaluated explanation
 ┌──────────────────────┐      ┌────────────────────────┐      ┌──────────────────────────┐
 │ risk_score: 0.89     │      │ generate explanation   │      │ member-friendly text     │
 │ risk_band: high      │  ──▶ │   → grade (4 evaluators)│ ──▶  │ + faithfulness/coverage/ │
 │ factors[]: name,     │      │   → rewrite or reject  │      │   readability/safety     │
 │   weight, evidence   │      │   → persist + audit    │      │ + decision + audit trail │
 └──────────────────────┘      └────────────────────────┘      └──────────────────────────┘
```

**Input** — a structured `RiskCase`:

```json
{
  "member_id": "mem_hea_001",
  "case_id": "healthcare_readmission_001",
  "risk_score": 0.89,
  "risk_band": "high",
  "model_name": "readmission_risk_v1",
  "model_version": "2026.01",
  "factors": [
    {
      "name": "Recent emergency visit",
      "direction": "increases_risk",
      "weight": 0.29,
      "evidence": "Member had two emergency visits in the last sixty days."
    },
    {
      "name": "Medication refill gap",
      "direction": "increases_risk",
      "weight": 0.18,
      "evidence": "A refill gap of twenty one days was detected for a prescribed medication."
    }
  ]
}
```

**Output** — a validated, grounded explanation (summary + per-factor plain-English reasons + limitations), each tied to the evidence it used, plus the four evaluation scores and a `passed`/`failed` decision.

## How it works (in detail)

The pipeline is orchestrated as an explicit **LangGraph workflow** — generation and each evaluator are discrete nodes, so retries and routing are visible and testable:

```text
load_case
  → generate_explanation          (LLM, strict JSON contract, 1 corrective retry)
  → evaluate_faithfulness
  → evaluate_coverage
  → evaluate_readability
  → evaluate_safety
      ├─ all pass            → persist_result        (status: passed)
      ├─ fail, retries left  → rewrite_explanation → re-evaluate
      └─ fail, retries spent → fail_gracefully       (status: failed, reasons recorded)
```

1. **Validate & persist.** A `RiskCase` is validated by Pydantic (`extra="forbid"` rejects stray fields — a guard against accidental PII) and stored in PostgreSQL.
2. **Generate.** The LLM produces an explanation against a **strict JSON contract** (`summary`, `member_explanation`, `factor_explanations[]` with the `source_evidence_used`, and `limitations[]`). Invalid JSON triggers exactly one corrective retry; a second failure rejects the generation.
3. **Evaluate.** The output runs through four deterministic evaluators (below). It is only marked `passed` if **all four** pass.
4. **Rewrite or fail.** If any evaluator fails and retries remain, the explanation is rewritten and re-evaluated. If retries are exhausted, the run fails gracefully with the reasons recorded.
5. **Persist & audit.** The explanation, scores, decision, and a details blob are stored — and **every run, pass or fail, writes an audit log**.

**The `fake` provider.** The LLM client is a thin text-in/text-out interface with three backends: `fake` (default), `openai`, and `anthropic`. The `fake` provider deterministically emits grounded JSON from the case in the prompt, so the **entire pipeline, test suite, and batch evaluation run offline with no API key**. Set `LLM_PROVIDER=openai` or `anthropic` (plus the matching key) for real generation.

## The evaluation harness

All four evaluators are **pure, deterministic functions** of `(ExplanationOutput, RiskCase)` — reproducible and unit-testable with no network calls. Deterministic checks come first by design; an LLM-as-judge can be layered on later but is never the only gate.

| Evaluator | What it checks | Method | Default pass threshold |
|-----------|----------------|--------|------------------------|
| **Faithfulness** | Every claim traces to the input | Each cited `source_evidence_used` must exist in the case (a fabricated citation = automatic fail); reasoning must share ≥ 30% content words with the grounding context; each member-explanation sentence must reference a known factor or overlap the context. Score = `supported / total claims`. | **0.85** |
| **Coverage** | The important factors were actually explained | Weight-aware: score = `covered_weight / total_weight`, so omitting a high-weight factor hurts more than a low-weight one. | **0.75** |
| **Readability** | Member-friendly, not technical | Flesch-Kincaid grade, average sentence length, and a jargon word list (e.g. *regression, coefficient, propensity*). | grade ≤ **12**, sentences ≤ 28 words, no jargon |
| **Safety** | No non-compliant wording | Conservative keyword/phrase rules for medical/legal advice, diagnosis, absolute guarantees, blame, and discriminatory language. Any match fails. False positives are preferred over false negatives. | any match = fail |

Results are flattened into an `eval_results` row (scores, `safety_passed`, `decision`, and a `details` blob with unsupported claims, missing factors, jargon hits, and failure reasons). See [`docs/eval_methodology.md`](docs/eval_methodology.md).

## Findings & measured metrics

> **Honesty rule:** these are the only numbers the implemented scripts actually produce. Every figure below is reproducible by re-running the batch evaluator. Nothing here is aspirational — see [`docs/resume_metrics.md`](docs/resume_metrics.md) and the explicit list of claims we *won't* make.

### Pipeline & evaluators — 60 synthetic cases (`fake` provider)

Measured with `python -m app.evals.run_batch --input sample_data/risk_cases.json`. Because the provider is the deterministic `fake` backend, these numbers measure **the pipeline and the evaluators**, not the quality of any specific hosted LLM.

| Metric | Value |
|--------|-------|
| Synthetic cases evaluated | 60 (5 domains × 12) |
| Generated successfully | 60 / 60 (100%) |
| **Passed all four evaluators** | **50 / 60 (83.3%)** |
| Average faithfulness score | **1.000** |
| Average coverage score | **1.000** |
| Average readability (FK grade) | **10.06** (range 6.9–13.8) |
| Unsupported-claim rate | **0.0%** |
| Expected-factor coverage | **100.0%** |
| Failure reasons | 10 readability (grade > 12) |

**Key finding:** the 10 readability failures are a *deliberate, positive signal*. The harness flagged explanations whose grade level exceeded the member-facing threshold — proving it **discriminates rather than rubber-stamps**. A harness that passes 100% of everything is not actually evaluating anything.

### Test coverage

- **71 automated tests** (unit + integration + schema validation), green in CI.
- Cover: schema validation, API routes, DB migrations + linkage, the LLM retry policy, all four evaluators, the LangGraph workflow (pass / retry / fail paths), dataset reproducibility, and the batch CLI.

## Real upstream risk model (end-to-end)

ModelLens explains *structured risk outputs* — and those outputs can come from a **real trained model**, not just synthetic JSON. The optional `app/risk_model/` package trains a logistic-regression credit-risk classifier on the public **German Credit** dataset (1,000 real applicants, no PII; CSV committed under `data/`), then converts each applicant's prediction into a `RiskCase`:

```bash
pip install -e ".[ml]"                  # scikit-learn + pandas

# Train the model and emit real risk cases (prints AUC / accuracy)
python -m app.risk_model.generate --n 50 --output sample_data/real_risk_cases.json

# Explain + evaluate those real cases through the same pipeline
python -m app.evals.run_batch --input sample_data/real_risk_cases.json
```

Each feature's `coefficient × standardized value` is an **exact, signed contribution**, which maps directly onto a factor's `direction` (increases/decreases risk) and normalized `weight`, with the feature value as the grounding `evidence`.

| Metric | Value |
|--------|-------|
| Dataset | German Credit, 1,000 applicants (750 train / 250 test) |
| Model | Logistic regression (exact coefficient × value attribution) |
| **ROC-AUC (held-out)** | **0.804** |
| **Accuracy (held-out)** | **0.772** |
| Real cases explained + evaluated | 50 |
| Pass rate through eval pipeline | 100% (faithfulness/coverage 1.000, readability grade 8.69) |

This demonstrates the complete path: **raw data → trained model → structured risk outputs → grounded, evaluated, audited explanations.** The package is deliberately isolated from the API serving path, so the API image does not depend on scikit-learn.

## Tech stack

| Layer | Choice | Why |
|-------|--------|-----|
| Language | **Python 3.11+** | Ecosystem for ML + LLM tooling |
| API | **FastAPI** + **Pydantic** | Typed request/response, automatic validation, `422` on bad input |
| Orchestration | **LangGraph** | Explicit stateful graph for generate → evaluate → rewrite/fail, with retries |
| LLM | **OpenAI / Anthropic** (pluggable; `fake` for offline) | Provider-agnostic, strict structured-JSON contract |
| Persistence | **PostgreSQL** + **SQLAlchemy 2.0** + **Alembic** | Relational integrity for case ↔ explanation ↔ eval linkage + migrations |
| ML (upstream) | **scikit-learn** + **pandas** | Real credit-risk model feeding the pipeline end-to-end |
| Quality | **Pytest** · **Ruff** | 71 tests + lint, enforced in CI |
| Ops | **Docker Compose** · **GitHub Actions** | One-command local stack; CI gate on every PR; branch protection on `main` |

## Architecture & data model

```text
Structured risk case (JSON)
        │  POST /cases
        ▼
   FastAPI + Pydantic validation
        │
        ▼
   PostgreSQL  (members, risk_cases, risk_factors,
                explanations, eval_results, audit_logs)
        │  POST /cases/{id}/explanations
        ▼
   LangGraph workflow  →  LLM layer (fake | openai | anthropic)
        │
        ▼
   Deterministic evaluation harness → eval_results + audit_log
```

```text
members ──< risk_cases ──< risk_factors
                  └──< explanations ──< eval_results
audit_logs  (reference case_id and/or explanation_id)
```

Every generated artifact links back to its source case, so an explanation and its evaluation are always traceable to the original structured input. All tunables are environment-driven (`app/core/config.py`): `DATABASE_URL`, `LLM_PROVIDER`/`LLM_MODEL`, and the evaluation thresholds. The Docker entrypoint runs `alembic upgrade head` before serving; tests run against SQLite with the same migrations, so CI needs no database service. Full detail in [`docs/architecture.md`](docs/architecture.md).

## Getting started

```bash
# Install dependencies (uses uv; pip works too)
uv venv
uv pip install -e ".[dev]"

# Copy environment template
cp .env.example .env

# Run the API
uvicorn app.main:app --reload

# Check health
curl http://localhost:8000/health
# -> {"status":"ok","version":"0.1.0"}
```

The default `LLM_PROVIDER=fake` runs the whole system with **no API key and no network**.

**Development:**

```bash
ruff check .     # lint
pytest           # 71 tests (unit + integration + schema validation)
```

**With Docker:**

```bash
docker compose up        # starts postgres + api (migrations run on startup)
curl http://localhost:8000/health
```

Postgres is published on host port **5433** (container 5432) to avoid clashing with a local Postgres.

**Batch evaluation:**

```bash
python -m app.evals.dataset                                   # (re)generate 60 synthetic cases
python -m app.evals.run_batch --input sample_data/risk_cases.json   # writes eval_results.json
```

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness probe |
| `POST` | `/cases` | Create (validate + persist) a structured risk case |
| `GET`  | `/cases/{case_id}` | Fetch a risk case |
| `GET`  | `/cases` | List cases (paginated) |
| `POST` | `/cases/{case_id}/explanations` | Run the LangGraph workflow to generate + evaluate an explanation |
| `GET`  | `/cases/{case_id}/explanations/latest` | Most recent explanation + eval summary |

A passing run returns `201` with `status="passed"` and the evaluation summary; a run that fails evaluation after retries returns `200` with `status="failed"` and the reasons in `evaluation.details`. Invalid payloads return `422`.

## Project layout

```text
app/
  api/         FastAPI routes (health, case CRUD, explanation generation)
  core/        config, logging, errors
  db/          SQLAlchemy 2.0 models, session, repository, Alembic migrations
  schemas/     Pydantic contracts (RiskCase, ExplanationOutput, eval, read models)
  llm/         provider clients, prompts, structured-output parsing with retry
  graph/       LangGraph state, nodes, workflow wiring
  evals/       deterministic evaluators, synthetic dataset generator, batch CLI
  risk_model/  real credit-risk model (train, attribution, RiskCase adapter)
tests/         unit + integration
data/          committed German Credit dataset (CSV)
sample_data/   synthetic + real-model risk cases
docs/          architecture, eval methodology, resume metrics
```

## Engineering practices

- **No direct pushes to `main`** — every change went through a feature branch → PR → CI → squash merge (the full project shipped as 14 reviewed PRs across phases 0–11).
- **Branch protection on `main`** requires both CI jobs (`Lint, test, schema validation` and `Docker build + compose smoke test`) to pass before merge; force-pushes and deletions are blocked.
- **CI on every PR** runs ruff, the full pytest suite, schema validation, a dataset reproducibility check, and a Docker Compose smoke test of the full create-case → generate-explanation flow.

## Safety & data policy

Synthetic data only, plus the public, PII-free German Credit dataset. **No PHI/PII**, no real patient, financial, or customer records, no SSNs or addresses. The LLM is constrained to explain only the provided structured input — it never diagnoses, gives medical or legal advice, claims causality not present in the input, or invents risk factors. The safety evaluator is intentionally conservative (false positives preferred over false negatives).

## Limitations & honest scope

To keep the metrics resume-safe, the following are **explicitly not claimed** because no implemented measurement supports them:

- ❌ Any "analyst review-time reduction %" — no human-time benchmark exists.
- ❌ Any "grounding accuracy %" — faithfulness here is a lexical baseline measured against the `fake` provider, not a labeled accuracy study.
- ❌ A "200-case eval set" — the dataset is 60 synthetic cases.
- ❌ Any latency claim implying real-LLM performance — measured latency reflects the deterministic `fake` provider.

When real LLM providers and labeled data are introduced, re-run the batch eval and update [`docs/resume_metrics.md`](docs/resume_metrics.md) with the new measured figures.

---

See [`docs/architecture.md`](docs/architecture.md), [`docs/eval_methodology.md`](docs/eval_methodology.md), and [`docs/resume_metrics.md`](docs/resume_metrics.md) for the deeper write-ups.
