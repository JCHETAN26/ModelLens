# ModelLens

> Convert structured risk model outputs into faithful, member-facing plain-English explanations — with automated grounding, safety, and faithfulness evaluation.

ModelLens takes structured risk model outputs (risk scores, contributing factors, evidence snippets, model metadata) and generates clear, member-friendly explanations. Every explanation is traceable to the structured input; the system is designed **not to hallucinate**. A LangGraph workflow generates each explanation and runs it through deterministic faithfulness, coverage, readability, and safety evaluators before persisting it.

## Status

Feature-complete against the phased build plan: validated case ingestion, a LangGraph generation + evaluation workflow, PostgreSQL persistence with audit trails, a deterministic evaluation harness, a synthetic dataset + batch eval, Docker Compose, and CI. See [`build-plan.md`](build-plan.md) for the roadmap and [`system-prompt.md`](system-prompt.md) for engineering rules.

## Tech stack

Python 3.11+ · FastAPI · Pydantic · LangGraph · PostgreSQL · SQLAlchemy · Alembic · Pytest · Ruff · Docker Compose · GitHub Actions

## Quick start

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

## Development

```bash
ruff check .     # lint
pytest           # tests
```

The default `LLM_PROVIDER=fake` lets the full pipeline and test suite run with **no API key and no network**, using a deterministic stub. Set `LLM_PROVIDER=openai` or `anthropic` with the matching API key for real generation.

## Run with Docker

```bash
docker compose up        # starts postgres + api (migrations run on startup)
curl http://localhost:8000/health
```

Postgres is published on host port **5433** (container 5432) to avoid clashing with a local Postgres.

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness probe |
| `POST` | `/cases` | Create (validate + persist) a structured risk case |
| `GET`  | `/cases/{case_id}` | Fetch a risk case |
| `GET`  | `/cases` | List cases (paginated) |
| `POST` | `/cases/{case_id}/explanations` | Run the LangGraph workflow to generate + evaluate an explanation |
| `GET`  | `/cases/{case_id}/explanations/latest` | Most recent explanation + eval summary |

A passing run returns `201` with `status="passed"` and the evaluation summary; a run that fails evaluation after retries returns `200` with `status="failed"` and the reasons in `evaluation.details`.

## How it works

1. A structured `RiskCase` is validated by Pydantic and persisted.
2. The **LangGraph workflow** generates an explanation via the LLM (strict JSON contract, one corrective retry), then runs four **deterministic evaluators** — faithfulness, coverage, readability, safety.
3. If all pass → persist. If any fail and retries remain → rewrite and re-evaluate. If retries are exhausted → fail gracefully. **Every run writes an audit log.**
4. Results (explanation + scores + decision) are stored and returned.

See [`docs/architecture.md`](docs/architecture.md) and [`docs/eval_methodology.md`](docs/eval_methodology.md).

## Synthetic dataset & batch evaluation

```bash
python -m app.evals.dataset                                   # (re)generate 60 synthetic cases
python -m app.evals.run_batch --input sample_data/risk_cases.json
```

Batch eval reports pass rate, average faithfulness/coverage/readability, latency, unsupported-claim rate, and a failure-reason histogram to `eval_results.json`. Latest measured numbers: [`docs/resume_metrics.md`](docs/resume_metrics.md).

## Real upstream risk model (end-to-end)

ModelLens explains *structured risk outputs* — and those outputs can come from a **real trained model**, not just synthetic data. `app/risk_model/` trains a logistic-regression credit-risk classifier on the public **German Credit** dataset (1,000 real applicants, no PII; CSV committed under `data/`), then turns each applicant's per-feature contributions into a `RiskCase`:

```bash
pip install -e ".[ml]"                  # scikit-learn + pandas

# Train the model and emit real risk cases (prints AUC / accuracy)
python -m app.risk_model.generate --n 50 --output sample_data/real_risk_cases.json

# Explain + evaluate those real cases through the same pipeline
python -m app.evals.run_batch --input sample_data/real_risk_cases.json
```

Each feature's `coefficient × standardized value` is an exact, signed contribution → it maps directly onto a factor's `direction` (increases/decreases risk) and `weight`, with the feature value as the grounding `evidence`. Measured: **ROC-AUC ≈ 0.80, accuracy ≈ 0.77** on a held-out split. This makes the system run front-to-back: **raw data → trained model → explained, evaluated, audited output.**

## Project layout

```
app/
  api/      FastAPI routes
  core/     config, logging, errors
  db/       SQLAlchemy models, session, migrations
  schemas/  Pydantic request/response models
  llm/      provider clients, prompts, structured output parsing
  graph/    LangGraph state, nodes, workflow
  evals/    evaluators, synthetic dataset generator, batch CLI
  risk_model/  real credit-risk model (train, attribution, RiskCase adapter)
tests/      unit + integration
data/       committed German Credit dataset (CSV)
sample_data/  synthetic + real-model risk cases
docs/       architecture, eval methodology, resume metrics
```

## Safety & data

Synthetic data only. No PHI/PII, no real patient, financial, or customer records. The LLM is constrained to explain only the provided structured input — it never diagnoses, gives medical/legal advice, or invents risk factors.
