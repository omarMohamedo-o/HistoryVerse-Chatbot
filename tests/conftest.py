"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from historyverse.api.app import create_app
from historyverse.api.dependencies import get_llm_provider
from historyverse.config.settings import Settings
from historyverse.core.base import BaseLLMProvider
from historyverse.core.enums import DeviceType, LLMProvider
from historyverse.services.memory.manager import ConversationMemory


class MockLLMProvider(BaseLLMProvider):
    """Predictable mock provider for unit and integration testing."""

    def __init__(self, model_name: str = "mock-model") -> None:
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    async def is_available(self) -> bool:
        return True

    async def generate(
        self,
        prompt: str,
        *,
        images: list[Image.Image] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> str:
        if "fail" in prompt.lower():
            return "I cannot answer this question."
        return "The Greco-Roman Museum in Alexandria houses ancient classical artifacts."


@pytest.fixture
def test_settings() -> Settings:
    """Fixture providing isolated test settings."""
    return Settings(
        llm_provider=LLMProvider.GEMINI,
        llm_model_name="mock-model",
        llm_temperature=0.2,
        llm_device_type=DeviceType.CPU,
        chroma_persist_dir="./test_DB",
        source_documents_dir="./test_dataset",
        drift_confidence_threshold=0.6,
        drift_max_retries=2,
    )


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """Fixture returning a mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def conversation_memory(test_settings: Settings) -> ConversationMemory:
    """Fixture providing a fresh conversation memory manager."""
    return ConversationMemory(test_settings)


@pytest.fixture
def client(mock_llm: MockLLMProvider) -> TestClient:
    """Fixture providing a FastAPI TestClient with mocked dependencies."""
    app = create_app()
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm
    return TestClient(app)
