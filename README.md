# ModelLens

**ModelLens turns the cold, numeric output of a risk model into a clear written explanation a normal person can understand — without making anything up.**

If a computer model decides someone is "high risk," ModelLens explains *why*, in plain language, using only the facts the model actually looked at. Then it automatically double-checks its own explanation before showing it to anyone.

---

## Table of contents

- [The problem, in plain terms](#the-problem-in-plain-terms)
- [What ModelLens does](#what-modellens-does)
- [Why this matters](#why-this-matters)
- [A real example](#a-real-example)
- [How it works, step by step](#how-it-works-step-by-step)
- [The four automatic checks](#the-four-automatic-checks)
- [What we found (results & metrics)](#what-we-found-results--metrics)
- [We tested it on real data, too](#we-tested-it-on-real-data-too)
- [Architecture diagram](#architecture-diagram)
- [What it's built with](#what-its-built-with)
- [Try it yourself](#try-it-yourself)
- [What's in the project](#whats-in-the-project)
- [Safety and data](#safety-and-data)
- [Honest about what we measured](#honest-about-what-we-measured)

---

## The problem, in plain terms

Banks, hospitals, and insurance companies use computer models to score people for *risk* — for example, "how likely is this patient to be readmitted to the hospital?" or "how likely is this applicant to default on a loan?"

These models spit out numbers: a score like `0.87`, a label like `high`, and a list of contributing factors with weights. That's useful to a data scientist, but it's meaningless to the person it's about — and even to the staff who have to act on it. Someone still has to sit down and **write out, in normal words, why the score came out the way it did.** That's slow, it's done differently by every person, and it's hard to check afterward.

The tempting shortcut is: "just let an AI chatbot write the explanation." But left unchecked, an AI will happily **invent reasons that were never in the data**, give medical or legal advice it shouldn't, or sound falsely certain. In healthcare or finance, that's not just sloppy — it's a real problem.

**ModelLens is the careful version of that shortcut.**

## What ModelLens does

You give ModelLens the structured output of a risk model — the score, the factors, and the evidence behind each factor. ModelLens then:

1. **Writes a plain-English explanation** of why the score is what it is.
2. **Checks itself** — automatically — to make sure the explanation is accurate, complete, easy to read, and safe.
3. **Rewrites or rejects** the explanation if it fails any of those checks. It never quietly hands back a bad answer.
4. **Keeps a record** of every explanation and every check, so you can always trace an answer back to the exact facts that produced it.

The whole point is trust: an explanation you can actually rely on, because the system proved it to itself before showing it to you.

## Why this matters

- **For the people being scored** — they get a clear reason instead of a mystery number.
- **For the staff** — explanations are written instantly and consistently, not by hand, one at a time.
- **For the organization** — every explanation is checked and logged, which is exactly what regulated industries (healthcare, insurance, finance) need.
- **The key guarantee** — ModelLens only ever explains things that are actually in the data. It is built so it *cannot* make up reasons.

## A real example

**What goes in** (the risk model's output for one person):

```json
{
  "risk_score": 0.89,
  "risk_band": "high",
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
      "evidence": "A refill gap of twenty-one days was detected for a prescribed medication."
    }
  ]
}
```

**What comes out** — a short, member-friendly explanation that says, in plain words, that this person is considered high risk mainly because of two recent emergency visits and a gap in refilling their medication — *and nothing the data doesn't support.* Alongside it come the automatic check scores and a clear pass/fail decision.

## How it works, step by step

Think of it as an assembly line where each station has one job:

1. **Take in the case.** The risk model's output is checked for completeness and saved. Anything malformed is rejected up front.
2. **Write the explanation.** The AI drafts an explanation, but it's strictly required to base every statement on the provided facts and to return its answer in a fixed format. If it returns something malformed, it gets exactly one chance to fix it.
3. **Run four checks** (described below).
4. **Decide what to do:**
   - All four checks pass → save the explanation and return it.
   - A check fails but there are tries left → **rewrite and check again.**
   - Out of tries → stop cleanly and record exactly why it failed. (No half-baked answer is ever returned.)
5. **Log everything.** Every run — success or failure — is written to an audit trail.

> One handy detail: ModelLens ships with a built-in "offline" mode that simulates the AI, so the entire system, including all the checks, runs on a laptop with **no internet and no AI account needed.** Plug in a real AI provider (like OpenAI or Anthropic) whenever you want.

## The four automatic checks

Before any explanation is accepted, it must pass all four of these. They're simple, rule-based checks — they give the same answer every time, so the results are predictable and testable.

| Check | The question it answers | How it decides |
|------|--------------------------|----------------|
| **Faithfulness** | "Is every statement actually backed by the data?" | Each claim must point to real evidence from the case. If it cites evidence that doesn't exist, it fails instantly. |
| **Coverage** | "Did it explain the factors that actually mattered?" | The bigger a factor's weight, the more it counts. Skipping an important factor hurts the score more than skipping a minor one. |
| **Readability** | "Can a normal person read this?" | Checks reading grade level, sentence length, and bans technical jargon. |
| **Safety** | "Did it stay out of trouble?" | Blocks medical/legal advice, diagnoses, blame, absolute guarantees, and discriminatory wording. When in doubt, it rejects. |

An explanation is only accepted if it passes **all four**.

## What we found (results & metrics)

> Every number here was produced by the project's own measurement script and can be reproduced by re-running it. Nothing is guessed or inflated.

We built a test set of **60 made-up cases** spanning five areas — hospital readmission, insurance claims, credit, fraud, and operations — and ran them all through ModelLens:

| What we measured | Result |
|------------------|--------|
| Cases processed | 60 |
| Explanations generated successfully | 60 out of 60 (100%) |
| **Passed all four checks** | **50 out of 60 (83%)** |
| Statements backed by the data | **100%** (zero made-up claims) |
| Important factors covered | **100%** |
| Failures | 10 cases were too hard to read (above a 12th-grade reading level) |

**The most important takeaway:** those 10 "failures" are actually good news. They prove the readability check is *doing its job* — it caught explanations that were too complicated and refused to pass them. A system that approves 100% of everything isn't really checking anything.

On top of that, the project has **71 automated tests** that run on every change, covering the whole pipeline end to end.

## We tested it on real data, too

ModelLens isn't just fed hand-written examples. We connected it to a **real, trained risk model** to prove the whole chain works front to back.

We took the public **German Credit dataset** — 1,000 real (anonymous, no personal info) loan applicants — and trained an actual credit-risk model on it. The model's reasoning for each applicant is then handed straight to ModelLens to explain.

| What we measured | Result |
|------------------|--------|
| Real applicants in the dataset | 1,000 |
| Model accuracy | **77%** |
| Model quality score (ROC-AUC, where 1.0 is perfect and 0.5 is a coin flip) | **0.80** |
| Real cases explained and checked | 10 / 10 passed |

This shows the complete journey: **raw data → a real trained model → a plain-English, fact-checked, logged explanation.**

## Architecture diagram

Here's the whole flow at a glance — from a risk model's output to a verified explanation:

```text
        A risk model's output
   (a score, the factors, the evidence)
                  │
                  ▼
        ┌───────────────────┐
        │  Check & save it  │   ← reject anything malformed up front
        └─────────┬─────────┘
                  ▼
        ┌───────────────────┐
        │ Write explanation │   ← AI drafts it, grounded only in the data
        └─────────┬─────────┘
                  ▼
        ┌───────────────────────────────────────┐
        │   Run the four automatic checks:       │
        │   faithfulness · coverage ·            │
        │   readability · safety                 │
        └─────────┬─────────────────────────────┘
                  ▼
            All four pass? ──── yes ──▶  Save it + return it ✓
                  │
                  no
                  │
        Tries left? ── yes ──▶ Rewrite, then check again ↺
                  │
                  no
                  ▼
        Stop cleanly, record exactly why it failed ✗

   (Every single run — pass or fail — is written to an audit log.)
```

And here's how the saved information links together, so any answer can be traced back to its source:

```text
  members ──< risk_cases ──< risk_factors
                   └──< explanations ──< evaluation_results
  audit_logs  (point back to the case and/or explanation)
```

## What it's built with

In short: standard, widely-used tools, chosen so the project is reliable and easy to run.

| Part | Tool | Why, in plain terms |
|------|------|---------------------|
| Core language | **Python** | The standard for AI and data work |
| Web service | **FastAPI** | Receives requests and validates them automatically |
| Workflow engine | **LangGraph** | Runs the "assembly line" of write → check → rewrite |
| The AI | **OpenAI / Anthropic** (or a built-in offline simulator) | Writes the actual explanations |
| Storage | **PostgreSQL** | A reliable database that keeps every case, explanation, and check linked together |
| Real model | **scikit-learn** | Trains the real credit-risk model on the German Credit data |
| Quality | **Pytest + Ruff** | 71 automated tests + code-style checks on every change |
| Packaging | **Docker** | Start the whole thing with one command |

## Try it yourself

You don't need an AI account or internet — the built-in offline mode runs the full system.

```bash
# 1. Install
uv venv
uv pip install -e ".[dev]"
cp .env.example .env

# 2. Start the service
uvicorn app.main:app --reload

# 3. Check it's alive
curl http://localhost:8000/health      # -> {"status":"ok","version":"0.1.0"}
```

Run all 60 test cases through it and see the scorecard:

```bash
python -m app.evals.run_batch --input sample_data/risk_cases.json
```

Or start everything (service + database) with one Docker command:

```bash
docker compose up
```

Want to see the real-data path? Train the credit model and run its output through ModelLens:

```bash
pip install -e ".[ml]"
python -m app.risk_model.generate --n 10 --output sample_data/real_risk_cases.json
python -m app.evals.run_batch --input sample_data/real_risk_cases.json
```

## What's in the project

```text
app/
  api/         the web endpoints you call
  llm/         talks to the AI (or the offline simulator)
  graph/       the write → check → rewrite "assembly line"
  evals/       the four automatic checks + the scorecard tool
  risk_model/  the real credit-risk model trained on German Credit data
  db/          the database setup and tables
  schemas/     the strict formats for inputs and outputs
tests/         71 automated tests
data/          the real German Credit dataset
sample_data/   the 60 example cases
docs/          deeper write-ups (architecture, evaluation, metrics)
```

For more depth, see [`docs/architecture.md`](docs/architecture.md), [`docs/eval_methodology.md`](docs/eval_methodology.md), and [`docs/resume_metrics.md`](docs/resume_metrics.md).

## Safety and data

- **No real personal data.** The examples are made up, and the German Credit dataset is public and anonymous — no names, addresses, medical records, or IDs.
- **The AI is kept on a leash.** It can only explain the facts it's given. It never diagnoses, never gives medical or legal advice, and never invents a reason that wasn't in the data.
- The safety check deliberately errs on the side of caution — it would rather wrongly flag a fine explanation than let a bad one through.

## Honest about what we measured

To keep this trustworthy, here are claims we deliberately **do not** make, because we haven't measured them:

- ❌ "Saves analysts X% of time" — we never ran a real time study.
- ❌ "X% grounding accuracy" — our faithfulness check is a solid baseline, not a formal accuracy study against labeled data.
- ❌ Any speed claims about a real AI provider — our timing was measured with the offline simulator.

When real AI providers and labeled data are added, we'll re-run the measurements and update the numbers. Honest metrics only.
