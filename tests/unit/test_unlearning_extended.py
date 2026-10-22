"""Unit tests for unlearning validator, registry filtering, and engine fallback paths."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from agent.core.enums import UnlearningStatus
from agent.core.schemas import UnlearningRequest, UnlearningValidationReport
from agent.services.unlearning.engine import UnlearningEngine
from agent.services.unlearning.registry import TombstoneRegistry
from agent.services.unlearning.validator import UnlearningValidator


@pytest.mark.asyncio
async def test_unlearning_validator_properties_and_arabic_probes():
    """Verify UnlearningValidator retraction_indicators property and Arabic probes formulation."""
    mock_llm = MagicMock()
    validator = UnlearningValidator(llm_provider=mock_llm, retriever=None)
    assert len(validator.retraction_indicators) > 0
    assert "no record" in validator.retraction_indicators or "retracted" in validator.retraction_indicators

    # Arabic probes
    arabic_probes = validator.generate_adversarial_probes("توت عنخ آمون", language="ar")
    assert len(arabic_probes) == 3
    assert any("توت عنخ آمون" in p for p in arabic_probes)


@pytest.mark.asyncio
async def test_unlearning_validator_rag_residuals_branches():
    """Verify check_rag_residuals handles None retriever, matching residual chunks, and exceptions."""
    mock_llm = MagicMock()
    # 1. Retriever is None
    v_none = UnlearningValidator(llm_provider=mock_llm, retriever=None)
    res_none = await v_none.check_rag_residuals("pharaoh", ["pharaoh"])
    assert res_none == 0

    # 2. Retriever returns matching docs
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        Document(page_content="The pharaoh ruled ancient Egypt."),
        Document(page_content="Unrelated text."),
    ]
    v_match = UnlearningValidator(llm_provider=mock_llm, retriever=mock_retriever)
    res_match = await v_match.check_rag_residuals("pharaoh", ["pharaoh"])
    assert res_match == 1

    # 3. Retriever raises exception
    mock_retriever.retrieve.side_effect = RuntimeError("Index error")
    res_exc = await v_match.check_rag_residuals("pharaoh", ["pharaoh"])
    assert res_exc == 0


def test_tombstone_registry_get_and_agent_filtering():
    """Verify TombstoneRegistry get method and agent-specific isolation filtering."""
    registry = TombstoneRegistry()
    rec = registry.register(
        target_concept="secret_tomb",
        forbidden_keywords=["secret_tomb", "forbidden_vault"],
        affected_agents=["agent_restricted_only"],
    )

    # Test get by concept_id
    fetched = registry.get(rec.concept_id)
    assert fetched is not None
    assert fetched.target_concept == "secret_tomb"

    # Test is_forbidden with non-affected agent (should return False)
    forbidden, _ = registry.is_forbidden("This contains secret_tomb", agent_id="agent_unaffected")
    assert forbidden is False

    # Test is_forbidden with affected agent (should return True)
    forbidden, concept = registry.is_forbidden("This contains secret_tomb", agent_id="agent_restricted_only")
    assert forbidden is True
    assert concept == "secret_tomb"


@pytest.mark.asyncio
async def test_unlearning_engine_sync_memory_fallback():
    """Verify UnlearningEngine falls back to sync purge_from_all_agents when async is not implemented."""
    mock_memory = MagicMock(spec=["purge_from_all_agents"])
    mock_memory.purge_from_all_agents.return_value = 3

    mock_llm = MagicMock()
    tombstones = TombstoneRegistry()
    engine = UnlearningEngine(
        llm_provider=mock_llm,
        memory_manager=mock_memory,
        tombstone_registry=tombstones,
        retriever=None,
    )

    dummy_report = UnlearningValidationReport(
        certificate_id="cert-123",
        target_concept="obsolete_dynasty",
        status=UnlearningStatus.FORGOTTEN_VERIFIED,
        probes_evaluated=3,
        leakage_rate=0.0,
        validation_passed=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    engine._validator.validate = AsyncMock(return_value=dummy_report)

    req = UnlearningRequest(target_concept="obsolete_dynasty", purge_memory=True, purge_rag=False)
    cert = await engine.unlearn(req)
    assert cert.memory_turns_purged == 3
    mock_memory.purge_from_all_agents.assert_called_once()
