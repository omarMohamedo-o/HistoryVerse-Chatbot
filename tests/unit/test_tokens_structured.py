"""Unit tests for token context management and Pydantic structured output."""

import pytest

from agent.core.schemas import StructuredHistoricalAnalysis
from tests.conftest import MockLLMProvider


def test_token_counting_and_context_budget():
    provider = MockLLMProvider("mock-gpt-4o")

    text = "The Library of Alexandria was one of the largest and most significant libraries of the ancient world."
    count = provider.count_tokens(text)
    assert count > 0

    window = provider.get_context_window()
    assert window >= 4096

    # Test fitting to context budget
    short_text = provider.fit_to_context_budget(text, max_tokens=100)
    assert short_text == text

    truncated_text = provider.fit_to_context_budget(text, max_tokens=3)
    assert "[Context truncated" in truncated_text


@pytest.mark.asyncio
async def test_structured_output_generation():
    provider = MockLLMProvider()

    prompt = "Analyze the Greco-Roman Museum of Alexandria"
    structured_result = await provider.generate_structured(prompt, StructuredHistoricalAnalysis)

    assert isinstance(structured_result, StructuredHistoricalAnalysis)
    assert structured_result.topic == "Alexandria Museum"
    assert structured_result.historical_period == "Greco-Roman"
    assert len(structured_result.key_entities) > 0
    assert structured_result.confidence_score >= 0.9


@pytest.mark.asyncio
async def test_base_provider_defaults_and_confidence():
    from unittest.mock import patch

    # Defaults on base provider
    provider = MockLLMProvider("gemini-1.5-pro")
    assert await provider.generate_image("A pyramid") is None
    assert provider.supports_video_generation is False
    assert await provider.generate_video("A pyramid timelapse") is None
    assert provider.get_context_window() == 1000000

    provider_mistral = MockLLMProvider("mistral-large")
    assert provider_mistral.get_context_window() == 32768

    provider_llama = MockLLMProvider("llama-3-8b")
    assert provider_llama.get_context_window() == 8192

    # Confidence estimation low-confidence penalties
    conf_low = provider.estimate_confidence(
        "Who built the Great Pyramid?",
        "I don't know the exact details.",
    )
    assert conf_low < 0.65

    # Token counting fallback when tiktoken is unavailable
    with patch.dict("sys.modules", {"tiktoken": None}):
        count_fallback = provider.count_tokens("Four score and seven years ago")
        assert count_fallback > 0

    # Structured JSON extraction with conversational preamble
    preamble_output = (
        "Certainly! Here is your requested JSON response:\n"
        '{"topic": "Ptolemaic Dynasty", "historical_period": "Hellenistic", '
        '"confidence_score": 0.95, "key_entities": [{"name": "Ptolemy I", "era": "Hellenistic", "significance": "Founder of Ptolemaic Dynasty"}], '
        '"summary": "Egyptian dynasty", "sources": []}'
    )
    parsed = provider._parse_structured_json(preamble_output, StructuredHistoricalAnalysis)
    assert parsed.topic == "Ptolemaic Dynasty"
