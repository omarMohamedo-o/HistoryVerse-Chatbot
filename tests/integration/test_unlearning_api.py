"""Integration tests for Unlearning and Structured Output API routes."""

from fastapi.testclient import TestClient


def test_unlearning_api_workflow(client: TestClient):
    # 1. Trigger selective unlearning of a concept
    forget_payload = {
        "target_concept": "Submerged Alien Ruins",
        "forbidden_keywords": ["alien ruins", "alien pyramid"],
        "purge_rag": False,
        "purge_memory": True,
        "reason": "discredited claim",
    }
    response = client.post("/v1/unlearning/forget", json=forget_payload)
    assert response.status_code == 200
    report = response.json()
    assert report["validation_passed"] is True
    assert report["status"] == "forgotten_verified"
    assert "certificate_id" in report

    # 2. List active tombstones
    tb_response = client.get("/v1/unlearning/tombstones")
    assert tb_response.status_code == 200
    tombstones = tb_response.json()
    assert len(tombstones) >= 1
    concept_id = tombstones[0]["concept_id"]

    # 3. Validate forgetting via re-check endpoint
    validate_payload = {
        "target_concept": "Submerged Alien Ruins",
        "forbidden_keywords": ["alien ruins"],
    }
    val_resp = client.post("/v1/unlearning/validate", json=validate_payload)
    assert val_resp.status_code == 200
    val_data = val_resp.json()
    assert val_data["validation_passed"] is True

    # 4. Remove tombstone
    del_resp = client.delete(f"/v1/unlearning/tombstones/{concept_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "removed"


def test_chat_structured_api_route(client: TestClient):
    payload = {"query": "Tell me about the classical Greco-Roman Museum"}
    response = client.post("/v1/chat/structured", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "topic" in data
    assert "historical_period" in data
    assert "key_entities" in data
    assert "verified_facts" in data
