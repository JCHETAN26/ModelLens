"""API tests for risk case CRUD endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def _payload(case_id: str = "case_001", member_id: str = "mem_123") -> dict:
    return {
        "member_id": member_id,
        "case_id": case_id,
        "risk_score": 0.87,
        "risk_band": "high",
        "model_name": "readmission_risk_v1",
        "model_version": "2026.01",
        "factors": [
            {
                "name": "Recent emergency visit",
                "direction": "increases_risk",
                "weight": 0.31,
                "evidence": "2 ER visits in the last 60 days.",
            }
        ],
    }


def test_create_case(client: TestClient) -> None:
    resp = client.post("/cases", json=_payload())
    assert resp.status_code == 201
    body = resp.json()
    assert body["case_id"] == "case_001"
    assert body["member_id"] == "mem_123"
    assert body["id"] > 0
    assert len(body["factors"]) == 1


def test_get_case(client: TestClient) -> None:
    client.post("/cases", json=_payload())
    resp = client.get("/cases/case_001")
    assert resp.status_code == 200
    assert resp.json()["case_id"] == "case_001"


def test_get_missing_case_returns_404(client: TestClient) -> None:
    resp = client.get("/cases/does_not_exist")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "case_not_found"


def test_duplicate_case_returns_409(client: TestClient) -> None:
    client.post("/cases", json=_payload())
    resp = client.post("/cases", json=_payload())
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "case_already_exists"


def test_invalid_payload_returns_422(client: TestClient) -> None:
    bad = _payload()
    bad["risk_score"] = 5.0  # out of [0, 1]
    resp = client.post("/cases", json=bad)
    assert resp.status_code == 422


def test_list_cases(client: TestClient) -> None:
    client.post("/cases", json=_payload("case_a", "mem_1"))
    client.post("/cases", json=_payload("case_b", "mem_2"))
    resp = client.get("/cases")
    assert resp.status_code == 200
    case_ids = {c["case_id"] for c in resp.json()}
    assert {"case_a", "case_b"}.issubset(case_ids)


def test_list_cases_pagination(client: TestClient) -> None:
    for i in range(3):
        client.post("/cases", json=_payload(f"case_{i}", f"mem_{i}"))
    resp = client.get("/cases", params={"limit": 2})
    assert resp.status_code == 200
    assert len(resp.json()) == 2
