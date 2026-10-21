"""Shared pytest fixtures."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from agent.api.app import create_app
from agent.api.dependencies import get_llm_provider
from agent.config.settings import Settings
from agent.core.base import BaseLLMProvider
from agent.core.enums import DeviceType, LLMProvider
from agent.services.memory.manager import ConversationMemory


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
        if "[MANDATORY STRUCTURED OUTPUT INSTRUCTION]" in prompt:
            return json.dumps(
                {
                    "topic": "Alexandria Museum",
                    "historical_period": "Greco-Roman",
                    "summary": "Houses ancient classical artifacts from Alexandria.",
                    "key_entities": [
                        {
                            "name": "Greco-Roman Museum",
                            "era": "Ptolemaic-Roman",
                            "significance": "Archaeological repository",
                        }
                    ],
                    "verified_facts": [
                        {
                            "claim": "Founded in 1892",
                            "is_verified": True,
                            "confidence": 1.0,
                            "sources": ["Museum Records"],
                        }
                    ],
                    "confidence_score": 0.98,
                }
            )
        if "[MANDATORY UNLEARNING DIRECTIVE]" in prompt:
            return "I do not have verified historical records regarding this topic, as the claim has been retracted."
        lower_p = prompt.lower()
        if "serapeum" in lower_p:
            return "The Serapeum of Alexandria was an ancient Greek temple dedicated to Serapis, built under Ptolemy III."
        if "pharos" in lower_p or "lighthouse" in lower_p:
            return "The Pharos Lighthouse of Alexandria was built on the island of Pharos to guide ships into the ancient harbor."
        if "library of alexandria" in lower_p:
            return "The Library of Alexandria was an ancient center of scholarship founded under the Ptolemaic dynasty."
        if "great pyramid" in lower_p and "alien" in lower_p:
            return "The Great Pyramid of Giza was built in the 26th century BC for Pharaoh Khufu by ancient Egyptian workers, not extraterrestrials."
        if "atlantis" in lower_p:
            return "I do not have verified historical records regarding a secret Atlantis tunnel, as this claim is unverified."
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
