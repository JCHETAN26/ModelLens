"""Tests for the batch evaluation CLI."""

from __future__ import annotations

import json
from pathlib import Path

from app.evals.dataset import build_dataset
from app.evals.run_batch import main, run_batch
from app.llm.clients import FakeLLMClient


def test_run_batch_on_dataset() -> None:
    cases, expected = build_dataset()
    report = run_batch(cases, client=FakeLLMClient(), expected=expected)

    assert report.total_cases == len(cases)
    assert report.generated == len(cases)  # fake provider always generates
    assert 0.0 <= report.pass_rate <= 1.0
    assert report.passed <= report.total_cases
    # Fake output echoes evidence, so faithfulness and coverage are perfect.
    assert report.avg_faithfulness == 1.0
    assert report.avg_coverage == 1.0
    assert report.avg_expected_factor_coverage == 1.0
    assert report.unsupported_claim_rate == 0.0


def test_run_batch_is_reproducible() -> None:
    cases, expected = build_dataset()
    a = run_batch(cases, client=FakeLLMClient(), expected=expected)
    b = run_batch(cases, client=FakeLLMClient(), expected=expected)
    # Latency varies, but the scored metrics must be identical.
    assert a.pass_rate == b.pass_rate
    assert a.avg_faithfulness == b.avg_faithfulness
    assert a.failure_reason_counts == b.failure_reason_counts


def test_main_writes_results(tmp_path: Path) -> None:
    cases, expected = build_dataset()
    input_path = tmp_path / "risk_cases.json"
    input_path.write_text(json.dumps(cases))
    (tmp_path / "expected_outputs.json").write_text(json.dumps(expected))
    output_path = tmp_path / "out.json"

    rc = main(["--input", str(input_path), "--output", str(output_path)])
    assert rc == 0

    saved = json.loads(output_path.read_text())
    assert saved["total_cases"] == len(cases)
    assert "outcomes" in saved
    assert len(saved["outcomes"]) == len(cases)
