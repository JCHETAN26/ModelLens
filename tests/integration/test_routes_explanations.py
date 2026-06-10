"""API tests for explanation generation endpoints (uses the fake LLM provider)."""

from __future__ import annotations

from fastapi.testclient import TestClient


def _case_payload(case_id: str = "case_x") -> dict:
    return {
        "member_id": "mem_x",
        "case_id": case_id,
        "risk_score": 0.8,
        "risk_band": "high",
        "model_name": "m",
        "model_version": "v1",
        "factors": [
            {
                "name": "Recent emergency visit",
                "direction": "increases_risk",
                "weight": 0.6,
                "evidence": "Member had two emergency visits in the last sixty days.",
            },
            {
                "name": "Medication refill gap",
                "direction": "increases_risk",
                "weight": 0.4,
                "evidence": "Refill gap of twenty one days detected.",
            },
        ],
    }


def test_generate_explanation(client: TestClient) -> None:
    client.post("/cases", json=_case_payload())
    resp = client.post("/cases/case_x/explanations")
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "passed"
    assert body["member_explanation"]
    assert len(body["factor_explanations"]) == 2
    assert body["evaluation"]["decision"] == "pass"
    assert 0.0 <= body["evaluation"]["faithfulness_score"] <= 1.0


def test_generate_explanation_missing_case_404(client: TestClient) -> None:
    resp = client.post("/cases/nope/explanations")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "case_not_found"


def test_get_latest_explanation(client: TestClient) -> None:
    client.post("/cases", json=_case_payload())
    client.post("/cases/case_x/explanations")
    resp = client.get("/cases/case_x/explanations/latest")
    assert resp.status_code == 200
    assert resp.json()["case_id"] == "case_x"
    assert resp.json()["status"] == "passed"


def test_get_latest_returns_most_recent(client: TestClient) -> None:
    client.post("/cases", json=_case_payload())
    first = client.post("/cases/case_x/explanations").json()
    second = client.post("/cases/case_x/explanations").json()
    latest = client.get("/cases/case_x/explanations/latest").json()
    assert second["id"] > first["id"]
    assert latest["id"] == second["id"]


def test_get_latest_no_explanation_404(client: TestClient) -> None:
    client.post("/cases", json=_case_payload())
    resp = client.get("/cases/case_x/explanations/latest")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "explanation_not_found"
