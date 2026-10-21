"""Unit tests for the Machine Unlearning framework and validation engine."""

import pytest

from agentverse.core.enums import UnlearningStatus
from agentverse.core.schemas import UnlearningRequest
from agentverse.services.unlearning.engine import UnlearningEngine
from agentverse.services.unlearning.registry import TombstoneRegistry
from agentverse.services.unlearning.validator import UnlearningValidator
from tests.conftest import MockLLMProvider


def test_tombstone_registry_lifecycle():
    registry = TombstoneRegistry()

    # Register tombstone
    tb = registry.register(
        target_concept="Atlantis under Alexandria",
        forbidden_keywords=["atlantis", "submerged alien pyramid"],
        reason="discredited pseudohistorical claim",
    )
    assert tb.concept_id.startswith("tombstone-")
    assert "atlantis" in tb.forbidden_keywords

    # Check detection
    forbidden, concept = registry.is_forbidden("Is there an Atlantis ruin near Pharos?")
    assert forbidden is True
    assert concept == "Atlantis under Alexandria"

    not_forbidden, _ = registry.is_forbidden("Tell me about the Pharos Lighthouse.")
    assert not_forbidden is False

    # Check unlearning guard prompt
    guard = registry.get_unlearning_guard_prompt()
    assert "MANDATORY UNLEARNING DIRECTIVE" in guard
    assert "Atlantis under Alexandria" in guard

    # Delete tombstone
    success = registry.delete(tb.concept_id)
    assert success is True
    assert len(registry.list_tombstones()) == 0


def test_unlearning_validator_leakage_detection():
    validator = UnlearningValidator(MockLLMProvider())

    # Case 1: Permitted refusal (explicit retraction phrase)
    refusal_response = (
        "I do not have verified historical records regarding Atlantis under Alexandria, "
        "as this claim has been retracted and discredited."
    )
    leaked, conf = validator.check_response_leakage(refusal_response, ["atlantis"])
    assert leaked is False
    assert conf < 0.2

    # Case 2: Affirmative leak (mentions forbidden fact affirmatively)
    leak_response = "Recent expeditions proved the Atlantis ruins exist beneath the eastern harbor."
    leaked, conf = validator.check_response_leakage(leak_response, ["atlantis"])
    assert leaked is True
    assert conf > 0.0

    # Case 3: Completely clean response
    clean_response = "The Greco-Roman Museum contains classical Ptolemaic statuary."
    leaked, conf = validator.check_response_leakage(clean_response, ["atlantis"])
    assert leaked is False
    assert conf == 0.0


@pytest.mark.asyncio
async def test_unlearning_engine_end_to_end():
    mock_llm = MockLLMProvider()
    engine = UnlearningEngine(llm_provider=mock_llm)

    request = UnlearningRequest(
        target_concept="Ptolemaic Alien Coin",
        forbidden_keywords=["alien coin", "ptolemaic alien"],
        purge_rag=False,
        purge_memory=True,
        reason="hallucinated coin artifact",
    )

    report = await engine.unlearn(request)

    assert report.validation_passed is True
    assert report.leakage_rate == 0.0
    assert report.status == UnlearningStatus.FORGOTTEN_VERIFIED
    assert report.certificate_id.startswith("CERT-UNLEARN-")
    assert len(report.probes) == 3
