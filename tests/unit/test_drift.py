"""Unit tests for the DriftDetector."""

import pytest

from agent.config.settings import Settings
from agent.core.enums import DriftAction
from agent.services.llm.drift import DriftDetector
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


@pytest.mark.asyncio
async def test_drift_detector_docker_rebuild_trigger(monkeypatch):
    provider = MockLLMProvider()
    settings = Settings(
        drift_confidence_threshold=0.99,
        drift_max_retries=1,
        drift_action=DriftAction.DOCKER_REBUILD,
        docker_rebuild_command="echo 'rebuild called'",
    )
    detector = DriftDetector(provider, settings=settings)

    rebuild_called = []
    monkeypatch.setattr(
        detector, "trigger_docker_rebuild", lambda: (rebuild_called.append(True), True)[1]
    )

    prompt = "Tell me something"
    _, report = await detector.generate_with_drift_protection(prompt)

    assert report.docker_rebuild_triggered is True
    assert len(rebuild_called) == 1


def test_drift_detector_trigger_reingest(monkeypatch):
    provider = MockLLMProvider()
    settings = Settings()
    detector = DriftDetector(provider, settings=settings)

    ingest_called = []

    class DummyIngestor:
        def __init__(self, _settings):
            pass

        def ingest(self):
            ingest_called.append(True)

    import agent.services.rag.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "DocumentIngestor", DummyIngestor)

    success = detector.trigger_reingest()
    assert success is True
    assert len(ingest_called) == 1


@pytest.mark.asyncio
async def test_drift_detector_additional_branches(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    primary = MagicMock()
    primary.model_name = "primary-model"
    # First call low conf, second call higher conf
    primary.generate_with_confidence = AsyncMock(
        side_effect=[("Low conf answer", 0.4), ("High conf answer", 0.9)]
    )

    settings = Settings(drift_confidence_threshold=0.7, drift_max_retries=1)
    detector = DriftDetector(primary, settings=settings)

    ans, report = await detector.generate_with_drift_protection("Prompt")
    assert ans == "High conf answer"
    assert report.confidence_score == 0.9

    # Fallback provider used
    primary.generate_with_confidence = AsyncMock(return_value=("Still low", 0.3))
    fallback = MagicMock()
    fallback.model_name = "fallback-model"
    fallback.generate_with_confidence = AsyncMock(return_value=("Fallback win", 0.85))

    detector_fb = DriftDetector(primary, fallback_provider=fallback, settings=settings)
    ans2, report2 = await detector_fb.generate_with_drift_protection("Prompt 2")
    assert ans2 == "Fallback win"
    assert report2.fallback_used is True

    # Fallback provider error
    fallback.generate_with_confidence = AsyncMock(side_effect=Exception("Fallback error"))
    ans3, _ = await detector_fb.generate_with_drift_protection("Prompt 3")
    assert ans3 == "Still low"

    # Rebuild exception
    monkeypatch.setattr("subprocess.Popen", MagicMock(side_effect=OSError("Boom")))
    assert detector.trigger_docker_rebuild() is False

    # Reingest exception
    import agent.services.rag.ingest as ingest_mod

    class BrokenIngestor:
        def __init__(self, _settings):
            raise RuntimeError("Ingest error")

    monkeypatch.setattr(ingest_mod, "DocumentIngestor", BrokenIngestor)
    assert detector.trigger_reingest() is False
