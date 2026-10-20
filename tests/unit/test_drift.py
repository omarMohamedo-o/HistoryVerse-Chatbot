"""Unit tests for the DriftDetector."""

import pytest
from historyverse.config.settings import Settings
from historyverse.core.enums import DriftAction
from historyverse.services.llm.drift import DriftDetector
from tests.conftest import MockLLMProvider


@pytest.mark.asyncio
async def test_drift_detector_clean_pass():
    provider = MockLLMProvider()
    settings = Settings(drift_confidence_threshold=0.5, drift_max_retries=1)
    detector = DriftDetector(provider, settings=settings)

    prompt = "Tell me about the Greco-Roman Museum in Alexandria"
    answer, report = await detector.generate_with_drift_protection(prompt)

    assert "Greco-Roman Museum" in answer
    assert report.confidence_score >= 0.5
    assert report.retries == 0
    assert report.fallback_used is False


@pytest.mark.asyncio
async def test_drift_detector_low_confidence_retries():
    provider = MockLLMProvider()
    settings = Settings(drift_confidence_threshold=0.99, drift_max_retries=2)
    detector = DriftDetector(provider, settings=settings)

    prompt = "Tell me something"
    _, report = await detector.generate_with_drift_protection(prompt)

    assert report.retries > 0
    assert report.action_taken in list(DriftAction)
