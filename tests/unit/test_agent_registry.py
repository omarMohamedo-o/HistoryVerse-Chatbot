"""Unit tests for SQLite persistent agent registry and custom agent lifecycle."""

from fastapi.testclient import TestClient

from agent.api.app import create_app
from agent.api.dependencies import get_app_settings, get_llm_provider
from agent.config.settings import Settings
from agent.services.agents.orchestrator import AgentOrchestrator
from agent.services.agents.registry_store import AgentRegistryStore
from tests.conftest import MockLLMProvider


def test_agent_registry_store_crud(tmp_path):
    db_file = str(tmp_path / "test_registry.db")
    store = AgentRegistryStore(db_path=db_file)

    # 1. Save agent
    saved = store.save_agent(
        agent_id="medical-diagnostic",
        name="Dr. Sarah",
        role="Clinical Consultant",
        domain="medicine",
        system_prompt="You are a clinical assistant.",
        provider_name="gemini",
        model_name="gemini-1.5-flash",
        temperature=0.2,
        metadata={"hospital": "General", "api_key": "supersecretkey"},
    )
    assert saved["agent_id"] == "medical-diagnostic"
    assert saved["name"] == "Dr. Sarah"
    assert saved["metadata"]["hospital"] == "General"
    # Secret must be masked!
    assert saved["metadata"]["api_key"] == "******"

    # 2. Get agent
    fetched = store.get_agent("medical-diagnostic")
    assert fetched is not None
    assert fetched["domain"] == "medicine"

    # 3. List agents
    all_agents = store.list_agents()
    assert len(all_agents) == 1
    assert all_agents[0]["agent_id"] == "medical-diagnostic"

    # 4. Update agent
    store.save_agent(
        agent_id="medical-diagnostic",
        name="Dr. Sarah Updated",
        role="Chief Medical Officer",
        domain="medicine",
        system_prompt="Updated clinical prompt.",
    )
    updated = store.get_agent("medical-diagnostic")
    assert updated["name"] == "Dr. Sarah Updated"
    assert updated["role"] == "Chief Medical Officer"

    # 5. Delete agent
    deleted = store.delete_agent("medical-diagnostic")
    assert deleted is True
    assert store.get_agent("medical-diagnostic") is None


def test_orchestrator_persistence_and_rehydrate(tmp_path):
    db_file = str(tmp_path / "orch_reg.db")
    store = AgentRegistryStore(db_path=db_file)
    settings = Settings(agent_registry_db_path=db_file)

    mock_llm = MockLLMProvider()

    # Orchestrator 1: creates an agent
    orch1 = AgentOrchestrator(settings=settings, store=store)
    orch1.create_custom_agent(
        agent_id="legal-advisor",
        name="Advocate Alex",
        role="Legal Consultant",
        domain="law",
        system_prompt="Analyze legal questions according to maritime regulations.",
        llm_provider=mock_llm,
    )
    assert orch1.get_agent("legal-advisor") is not None

    # Orchestrator 2 (simulating fresh server restart): loads from same store
    store2 = AgentRegistryStore(db_path=db_file)
    orch2 = AgentOrchestrator(settings=settings, store=store2)
    # The agent must have been rehydrated on startup!
    rehydrated = orch2.get_agent("legal-advisor")
    assert rehydrated is not None
    assert rehydrated.name == "Advocate Alex"
    assert rehydrated.domain == "law"


def test_api_custom_agents_crud_and_rehydrate(tmp_path):
    db_file = str(tmp_path / "api_reg.db")
    settings = Settings(agent_registry_db_path=db_file)
    mock_llm = MockLLMProvider()

    app = create_app()
    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm

    client = TestClient(app)

    # 1. Create custom agent via API
    resp_create = client.post(
        "/v1/agents/custom",
        json={
            "agent_id": "crypto-analyst",
            "name": "Satoshi",
            "role": "Blockchain Architect",
            "domain": "finance",
            "system_prompt": "You are a blockchain systems researcher.",
        },
    )
    assert resp_create.status_code == 200
    assert resp_create.json()["agent_id"] == "crypto-analyst"

    # 2. Get agent details
    resp_get = client.get("/v1/agents/crypto-analyst")
    assert resp_get.status_code == 200
    assert resp_get.json()["name"] == "Satoshi"
    assert resp_get.json()["domain"] == "finance"

    # 3. Update agent
    resp_put = client.put(
        "/v1/agents/crypto-analyst",
        json={"role": "Lead Cryptographic Architect", "domain": "cryptography"},
    )
    assert resp_put.status_code == 200
    assert resp_put.json()["role"] == "Lead Cryptographic Architect"
    assert resp_put.json()["domain"] == "cryptography"

    # 4. Delete agent
    resp_del = client.delete("/v1/agents/crypto-analyst")
    assert resp_del.status_code == 200
    assert resp_del.json()["status"] == "deleted"

    # 5. Verify deleted
    resp_get_after = client.get("/v1/agents/crypto-analyst")
    assert resp_get_after.status_code == 404
