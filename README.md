# ModelLens

> Convert structured risk model outputs into faithful, member-facing plain-English explanations — with automated grounding, safety, and faithfulness evaluation.

ModelLens takes structured risk model outputs (risk scores, contributing factors, evidence snippets, model metadata) and generates clear, member-friendly explanations. Every explanation is traceable to the structured input; the system is designed **not to hallucinate**. A LangGraph workflow generates each explanation and runs it through deterministic faithfulness, coverage, readability, and safety evaluators before persisting it.

## Status

Early-stage build. See [`build-plan.md`](build-plan.md) for the phased roadmap and [`system-prompt.md`](system-prompt.md) for engineering rules.

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

## Project layout

```
app/
  api/      FastAPI routes
  core/     config, logging, errors
  db/       SQLAlchemy models, session, migrations
  schemas/  Pydantic request/response models
  llm/      provider clients, prompts, structured output parsing
  graph/    LangGraph state, nodes, workflow
  evals/    faithfulness, coverage, readability, safety evaluators
tests/      unit + integration
sample_data/  synthetic risk cases
docs/       architecture, eval methodology, resume metrics
```

## Safety & data

Synthetic data only. No PHI/PII, no real patient, financial, or customer records. The LLM is constrained to explain only the provided structured input — it never diagnoses, gives medical/legal advice, or invents risk factors.
