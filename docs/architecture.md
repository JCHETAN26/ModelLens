# ModelLens — Architecture

ModelLens converts **structured risk model outputs** into **faithful, member-facing explanations**, with every explanation evaluated for grounding, coverage, readability, and safety before it is persisted.

## High-level flow

```
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
   LangGraph workflow
   load_case → generate_explanation
   → evaluate_faithfulness → evaluate_coverage
   → evaluate_readability → evaluate_safety
        ├─ all pass            → persist_result
        ├─ fail, retries left  → rewrite_explanation → re-evaluate
        └─ fail, retries spent → fail_gracefully
        │
        ▼
   LLM layer (fake | openai | anthropic) — strict JSON contract
        │
        ▼
   Evaluation harness (deterministic) → eval_results + audit_log
```

## Components

| Layer | Module | Responsibility |
|-------|--------|----------------|
| API | `app/api/` | FastAPI routes: health, case CRUD, explanation generation |
| Schemas | `app/schemas/` | Pydantic contracts — input `RiskCase`, LLM `ExplanationOutput`, eval results, read models |
| DB | `app/db/` | SQLAlchemy 2.0 models, session, repository, Alembic migrations |
| LLM | `app/llm/` | Provider-agnostic clients, prompts, strict JSON parsing with retry |
| Graph | `app/graph/` | LangGraph state, nodes, workflow wiring |
| Evals | `app/evals/` | Deterministic evaluators, dataset generator, batch CLI |

## Data model

```
members ──< risk_cases ──< risk_factors
                  └──< explanations ──< eval_results
audit_logs  (reference case_id and/or explanation_id)
```

Every generated artifact links back to its source case, so an explanation and
its evaluation are always traceable to the original structured input. The
`extra="forbid"` setting on input schemas rejects unexpected fields (a guard
against stray PII).

## LLM layer & the `fake` provider

The LLM client is a thin text-in / text-out interface with three providers:
`fake` (default), `openai`, and `anthropic`. The **`fake` provider** reads the
structured case embedded in the prompt and emits deterministic, grounded JSON —
echoing each factor's evidence into its explanation. This lets the **entire
pipeline, test suite, and batch evaluation run offline with no API key**, while
real providers can be enabled via `LLM_PROVIDER` + the matching key.

Output is parsed against the strict `ExplanationOutput` schema. Invalid output
triggers exactly one corrective retry; a second failure rejects the generation.

## Workflow & retries

The LangGraph workflow is the orchestration backbone. Generation and the four
evaluators are explicit nodes. Routing after evaluation:

- **all evaluators pass** → `persist_result` (status `passed`)
- **any fail & `retry_count < MAX_REWRITE_RETRIES`** → `rewrite_explanation` → re-evaluate
- **any fail & retries exhausted** → `fail_gracefully` (status `failed`, reasons recorded)

Generation/validation failures short-circuit to `fail_gracefully`. **Every run
writes an audit log**, pass or fail.

## Configuration

All tunables are environment-driven (`app/core/config.py`): `DATABASE_URL`,
`LLM_PROVIDER`/`LLM_MODEL`, and evaluation thresholds
(`FAITHFULNESS_THRESHOLD`, `COVERAGE_THRESHOLD`, `READABILITY_MAX_GRADE`,
`MAX_REWRITE_RETRIES`). See `.env.example`.

## Upstream risk model (`app/risk_model/`)

ModelLens explains *structured risk outputs* — which can come from a real model.
This optional package (requires the `ml` extra) trains a logistic-regression
classifier on the public **German Credit** dataset and converts each applicant's
prediction into a `RiskCase`:

- `dataset.py` — loads the committed CSV (`data/german_credit.csv`); readable feature names.
- `train.py` — `ColumnTransformer` (standardize numeric, one-hot categorical) + `LogisticRegression`; reports held-out ROC-AUC / accuracy.
- `adapter.py` — per-feature contribution = `coefficient × encoded value` (exact, signed) → factor `direction` + normalized `weight`; feature value → `evidence`.
- `generate.py` — emits real `RiskCase`s consumable by the explanation pipeline and batch evaluator.

It is deliberately isolated from the API serving path, so the API image does
not depend on scikit-learn.

## Local deployment

`docker compose up` starts `postgres` (16) and `api`. The API container's
entrypoint runs `alembic upgrade head` before serving, so the schema is always
current. Tests run against SQLite (applying the same migrations), so CI needs no
database service.
