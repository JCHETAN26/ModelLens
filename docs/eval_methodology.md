# ModelLens — Evaluation Methodology

Evaluation is treated as a first-class part of the system, not an afterthought.
Every generated explanation passes through four **deterministic** evaluators
before it can be persisted as `passed`. Deterministic checks come first by
design; an LLM-as-judge may be layered on later but is never the only gate.

All evaluators are pure functions of `(ExplanationOutput, RiskCase)`, so results
are reproducible and unit-testable without any network calls.

## 1. Faithfulness (`app/evals/faithfulness.py`)

**Goal:** every claim in the explanation must be traceable to the structured input.

**Method (lexical-overlap baseline):**
- The grounding context is all structured input the explanation may draw on:
  factor evidence strings, factor names, and the risk band.
- Each **factor explanation** is a claim. Its cited `source_evidence_used` must
  exist in the case (otherwise it is a **fabricated citation**), and its
  reasoning must share ≥ 30% of its content words with the grounding context.
- Each **sentence** of the member explanation must either reference a known
  factor name or overlap the grounding context.

**Score:** `supported_claims / total_claims`. A fabricated citation forces a
fail regardless of score. **Pass threshold: 0.85** (configurable).

## 2. Coverage (`app/evals/coverage.py`)

**Goal:** the important factors must actually be explained.

**Method (weight-aware):** a factor is *covered* if it is explained or named in
the prose. The score is the **fraction of total factor weight that is covered**,
so omitting a high-weight factor lowers the score more than omitting a
low-weight one.

**Score:** `covered_weight / total_weight`. **Pass threshold: 0.75** (configurable).

## 3. Readability (`app/evals/readability.py`)

**Goal:** the explanation should be member-friendly, not technical.

**Method:** Flesch-Kincaid grade level over the summary + member explanation,
plus average sentence length and a jargon word list (e.g. *regression*,
*coefficient*, *propensity*, *quantile*).

**Pass condition:** grade ≤ 12, avg sentence length ≤ 28 words, and no jargon
terms. The reported `readability_score` is the grade level (lower is better).

## 4. Safety (`app/evals/safety.py`)

**Goal:** reject non-compliant wording.

**Method:** conservative keyword/phrase rules covering medical advice,
diagnosis, legal advice, absolute guarantees, blame-oriented language, and
discriminatory phrasing. Any match fails the explanation. False positives are
preferred over false negatives here.

## Aggregation & persistence

`runner.evaluate_explanation` runs all four and returns an `EvaluationResult`.
The result is `passed` only if **all four** pass. `to_record_fields()` flattens
it into the `eval_results` row (scores, `safety_passed`, `decision`, and a
`details` blob with unsupported claims, missing factors, jargon, safety flags,
and failure reasons).

## Batch evaluation

`python -m app.evals.run_batch --input sample_data/risk_cases.json` runs the
whole dataset and reports pass rate, average scores, latency, unsupported-claim
rate, expected-factor coverage, and a failure-reason histogram. Output is
written to `eval_results.json`. With the deterministic `fake` provider the
numbers are fully reproducible (asserted by a test).

See [`resume_metrics.md`](resume_metrics.md) for the latest **measured** numbers.
